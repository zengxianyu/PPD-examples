import os
import torch
import warnings
import torchvision
import hashlib
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from datasets import load_dataset


class HuggingFaceURLImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from HuggingFace datasets with URL fields.
    Implements lazy downloading with caching (Option 2).
    
    Note on num_workers:
        - num_workers=0: Recommended for first epoch (when downloading images)
        - num_workers>0: Can be used, but may have issues:
          * Race condition: If two workers check cache for the same URL simultaneously,
            both might see "file doesn't exist" and both download (wasteful but safe)
          * Duplicate URLs: If dataset has same URL at different indices, different workers
            might download the same image
          * Dataset repetition: If repeat > 1, same sample appears multiple times
          * After caching, subsequent epochs work well with multiple workers
        - Best practice: Use num_workers=0 for first epoch, then increase for later epochs
    
    Usage:
        dataset = HuggingFaceURLImageDataset(
            dataset_name="bghira/photo-concept-bucket",
            url_field="url",
            text_field="cogvlm_caption",  # or "description", "alt", etc.
            cache_dir="./url_cache",
            max_pixels=1920*1080,
            height=None,
            width=None,
        )
    """
    
    def __init__(
        self,
        dataset_name=None,
        dataset=None,
        dataset_config=None,
        split="train",
        url_field="url",
        text_field="cogvlm_caption",
        max_pixels=1920*1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        data_file_keys=("image",),
        cache_dir="./url_cache",
        repeat=1,
        timeout=10,
        max_samples=None,
        args=None,
    ):
        """
        Args:
            dataset_name: Name of the HuggingFace dataset (e.g., "bghira/photo-concept-bucket")
            dataset: Pre-loaded HuggingFace dataset object (alternative to dataset_name)
            dataset_config: Optional config name for the dataset
            split: Dataset split to use (default: "train")
            url_field: Field name containing the image URL
            text_field: Field name containing the text/prompt (default: "cogvlm_caption")
            max_pixels: Maximum pixels for dynamic resolution
            height: Fixed height (None for dynamic)
            width: Fixed width (None for dynamic)
            height_division_factor: Height must be divisible by this
            width_division_factor: Width must be divisible by this
            data_file_keys: Keys to include in returned data dict
            cache_dir: Directory to cache downloaded images
            repeat: Number of times to repeat the dataset per epoch
            timeout: Timeout for URL requests in seconds
            max_samples: Maximum number of samples to use (None for all, useful for debugging)
            args: Optional argparse args object (for compatibility with existing code)
        """
        if args is not None:
            # Support for argparse-based initialization
            dataset_name = getattr(args, 'dataset_name', dataset_name)
            url_field = getattr(args, 'url_field', url_field)
            text_field = getattr(args, 'text_field', text_field)
            cache_dir = getattr(args, 'cache_dir', cache_dir)
            height = getattr(args, 'height', height)
            width = getattr(args, 'width', width)
            max_pixels = getattr(args, 'max_pixels', max_pixels)
            data_file_keys = getattr(args, 'data_file_keys', data_file_keys)
            if isinstance(data_file_keys, str):
                data_file_keys = data_file_keys.split(",")
            repeat = getattr(args, 'dataset_repeat', repeat)
            max_samples = getattr(args, 'max_samples', max_samples)
        
        # Load dataset if not provided
        if dataset is None:
            if dataset_name is None:
                raise ValueError("Either dataset_name or dataset must be provided")
            self.dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
            )
        else:
            self.dataset = dataset
        
        # Limit dataset size for debugging if specified
        if max_samples is not None and max_samples > 0:
            original_size = len(self.dataset)
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            print(f"Limited dataset to {len(self.dataset)} samples (from {original_size}) for debugging")
        
        self.url_field = url_field
        self.text_field = text_field
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys if isinstance(data_file_keys, tuple) else tuple(data_file_keys)
        self.repeat = repeat
        self.timeout = timeout
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine resolution mode
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        else:
            raise ValueError("Both height and width must be specified, or both must be None")
        
        print(f"Dataset loaded: {len(self.dataset)} samples")
        print(f"Cache directory: {self.cache_dir}")
        print(f"URL field: {self.url_field}, Text field: {self.text_field}")
    
    def _get_cache_path(self, url):
        """Generate cache filename from URL using hash"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        # Try to get extension from URL
        ext = "jpg"  # default
        try:
            # Extract extension from URL (handle query parameters)
            url_clean = url.split('?')[0]
            if '.' in url_clean:
                ext = url_clean.split('.')[-1].lower()
                # Validate extension
                if ext not in ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp']:
                    ext = "jpg"
        except:
            pass
        return self.cache_dir / f"{url_hash}.{ext}"
    
    def load_image_from_url(self, url):
        """
        Load image from URL with caching.
        
        Note: With multiple workers, the same image might be downloaded if:
        1. Race condition (TOCTOU): Worker A checks cache (file doesn't exist), starts downloading.
           Worker B checks cache at nearly the same time (file still doesn't exist), also starts downloading.
           This is rare but can happen. threading.Lock doesn't help across processes.
        2. Duplicate URLs in dataset: Different indices might have the same URL.
        3. Dataset repetition: If repeat > 1, same sample appears multiple times.
        
        In practice, this is usually not a big problem - the second download will just overwrite
        the cache file (which is fine), and subsequent accesses will use the cached version.
        """
        cache_path = self._get_cache_path(url)
        
        # Check cache first
        if cache_path.exists():
            try:
                image = Image.open(cache_path).convert("RGB")
                return image
            except Exception as e:
                warnings.warn(f"Failed to load cached image {cache_path}: {e}. Re-downloading...")
                try:
                    cache_path.unlink()  # Remove corrupted cache file
                except:
                    pass
        
        # Download and cache
        # Note: There's a small race condition window here - if two processes check cache
        # simultaneously and both see file doesn't exist, both will download.
        # This is generally acceptable - the second write will just overwrite the first.
        try:
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Load image
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Save to cache
            try:
                image.save(cache_path, quality=95)
            except Exception as e:
                warnings.warn(f"Failed to cache image {url}: {e}")
            
            return image
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Failed to download image from {url}: {e}")
            return None
        except Exception as e:
            warnings.warn(f"Failed to process image from {url}: {e}")
            return None
    
    def crop_and_resize(self, image, target_height, target_width):
        """Crop and resize image to target dimensions"""
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        """Calculate target height and width based on image size and constraints"""
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def load_image(self, url):
        """Load and process image from URL"""
        image = self.load_image_from_url(url)
        if image is None:
            return None
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    def __getitem__(self, data_id):
        """Get a single data sample"""
        # Handle dataset repetition
        idx = data_id % len(self.dataset)
        item = self.dataset[idx]
        
        # Extract URL and text
        url = item.get(self.url_field)
        if url is None:
            warnings.warn(f"Missing URL field '{self.url_field}' at index {idx}")
            return None
        
        # Load image
        image = self.load_image(url)
        if image is None:
            return None
        
        # Build data dictionary
        data = {}
        
        # Add image if requested
        if "image" in self.data_file_keys:
            data["image"] = image
        
        # Add text if requested and available
        if "text" in self.data_file_keys or "prompt" in self.data_file_keys:
            text = item.get(self.text_field, "")
            if "text" in self.data_file_keys:
                data["text"] = text
            if "prompt" in self.data_file_keys:
                data["prompt"] = text
        
        # Add other fields if they're in data_file_keys
        for key in self.data_file_keys:
            if key not in data and key in item:
                data[key] = item[key]
        
        return data
    
    def __len__(self):
        """Return dataset length accounting for repetition"""
        return len(self.dataset) * self.repeat
    
    def preload_cache(self, max_samples=None, num_workers=4):
        """
        Pre-download and cache images (optional, for faster training start)
        
        Args:
            max_samples: Maximum number of samples to preload (None for all)
            num_workers: Number of parallel download workers
        """
        from concurrent.futures import ThreadPoolExecutor
        
        num_samples = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)
        print(f"Preloading {num_samples} images to cache...")
        
        def download_one(idx):
            item = self.dataset[idx]
            url = item.get(self.url_field)
            if url:
                self.load_image_from_url(url)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(
                executor.map(download_one, range(num_samples)),
                total=num_samples,
                desc="Caching images"
            ))
        
        print(f"Cache preloading complete!")

