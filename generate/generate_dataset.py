# -*- coding: utf-8 -*-
"""
CAPTCHA Dataset Generator - generates captcha images (png) with label files (txt)

Generation pipeline:
1. Initialization: load config, set random seed, init fonts/colors/interference params
2. Per-sample generation: random label, background (gradient/texture), characters,
   interference (lines, noise, adhesion), distortion (rotation, warp, perspective)
3. Quality check: blur (Laplacian), contrast (std dev), occlusion analysis, filtering

Output format:
- Image file: {uuid}.png
- Label file: {uuid}.txt (single-line label string)
"""

import logging
import os
import random
import uuid
import cv2
import numpy as np
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

from utils.common import PathValidator
from utils.config_loader import get_config


class CaptchaGenerator:
    """CAPTCHA image generator."""

    def __init__(self, output_dir, num_samples=20000, image_width=None, image_height=None, verbose=True, use_multiprocessing=False):
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing

        config = get_config()
        chars_config = config.get_chars_config()
        self._gen_config = config.get('captcha_generation', {}) or {}

        # Quality thresholds from config.yaml (dataset_cleaning.thresholds as source of truth)
        cleaning_config = config.get('dataset_cleaning', {}) or {}
        thresholds = cleaning_config.get('thresholds', {}) if isinstance(cleaning_config, dict) else {}
        self._quality_config = {
            'occlusion_threshold': 0.0,
            'truncation_margin': 0,
            'blur_threshold': float(thresholds.get('min_laplacian_var', 30)),
            'contrast_threshold': float(thresholds.get('min_contrast_std', 15)),
            'entropy_threshold': float(thresholds.get('min_entropy', 0.0))
        }

        self.characters = chars_config['characters']
        self.num_classes = chars_config['num_classes']
        self.max_length = chars_config['max_length']

        if image_width is None or image_height is None:
            original_width, original_height = config.get_original_image_size()
            self.image_width = image_width if image_width is not None else original_width
            self.image_height = image_height if image_height is not None else original_height
        else:
            self.image_width = image_width
            self.image_height = image_height

        try:
            PathValidator.validate_path(output_dir, "Output directory", must_exist=False)
            os.makedirs(output_dir, exist_ok=True)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error: {e}")
            raise

        self._font_pool = self._load_font_pool()

    def _get_range(self, key_path, default):
        value = self._gen_config
        for k in key_path.split('.'):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value if value is not None else default

    def _rand_int(self, key_path, default_min, default_max):
        v = self._get_range(key_path, [default_min, default_max])
        return random.randint(int(v[0]), int(v[1]))

    def _rand_float(self, key_path, default_min, default_max):
        v = self._get_range(key_path, [default_min, default_max])
        return random.uniform(float(v[0]), float(v[1]))

    def _rand_choice(self, key_path, default):
        v = self._get_range(key_path, default)
        if isinstance(v, list) and v:
            return random.choice(v)
        return v

    def generate_gradient_background(self, width, height):
        """Generate a light gradient background."""
        color_range = self._get_range('background.gradient_color_range', [160, 255])
        lo, hi = int(color_range[0]), int(color_range[1])
        lo = max(0, min(255, lo))
        hi = max(0, min(255, hi))
        if hi <= lo:
            hi = min(255, lo + 1)
        color1 = np.random.randint(lo, hi + 1, 3)
        color2 = np.random.randint(lo, hi + 1, 3)

        # Create gradient background
        background = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(width):
            alpha = i / width
            color = (1 - alpha) * color1 + alpha * color2
            background[:, i] = color.astype(np.uint8)

        return background

    def add_random_texture(self, image):
        """Add light random texture."""
        height, width = image.shape[:2]

        num_blocks = self._rand_int('background.num_color_blocks', 2, 6)
        alpha_min, alpha_max = self._get_range('background.alpha_range', [0.10, 0.28])
        for _ in range(num_blocks):
            x = random.randint(0, width - 20)
            y = random.randint(0, height - 20)
            w = random.randint(10, 30)
            h = random.randint(10, 30)
            color = tuple(np.random.randint(120, 256, 3).tolist())
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = random.uniform(float(alpha_min), float(alpha_max))
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        num_lines = self._rand_int('background.texture_lines', 8, 18)
        for _ in range(num_lines):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            color = tuple(np.random.randint(120, 256, 3).tolist())
            thickness = random.randint(1, 2)
            overlay = image.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
            alpha = random.uniform(float(alpha_min), float(alpha_max))
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image

    def generate_complex_background(self, width, height):
        background = self.generate_gradient_background(width, height)
        background = self.add_random_texture(background)

        num_blobs = self._rand_int('background.num_color_blobs', 3, 8)
        alpha_min, alpha_max = self._get_range('background.alpha_range', [0.10, 0.28])
        for _ in range(num_blobs):
            radius = random.randint(10, max(12, min(width, height) // 2))
            cx = random.randint(0, width - 1)
            cy = random.randint(0, height - 1)
            color = tuple(np.random.randint(80, 236, 3).tolist())
            overlay = background.copy()
            cv2.circle(overlay, (cx, cy), radius, color, -1)
            alpha = random.uniform(float(alpha_min), float(alpha_max))
            background = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)

        blur_k = self._get_range('background.blur_kernel', [1, 3])
        k = random.choice([int(blur_k[0]), int(blur_k[1])])
        k = max(1, int(k))
        if k % 2 == 0:
            k += 1
        if k > 1:
            background = cv2.GaussianBlur(background, (k, k), 0)

        return background

    def add_interference_lines(self, image):
        """Add multiple interference lines with random thickness, color, and curvature."""
        height, width = image.shape[:2]

        num_lines = self._rand_int('interference_lines.num_lines', 3, 7)
        thick_min, thick_max = self._get_range('interference_lines.thickness', [1, 3])
        alpha_min, alpha_max = self._get_range('interference_lines.alpha_range', [0.20, 0.55])
        color_min, color_max = self._get_range('interference_lines.color_range', [30, 220])
        pmin, pmax = self._get_range('interference_lines.curve_points', [5, 9])

        overlay = image.copy()
        for _ in range(num_lines):
            x1 = random.randint(-10, width + 10)
            x2 = random.randint(-10, width + 10)
            y1 = random.randint(int(height * 0.15), int(height * 0.85))
            y2 = random.randint(int(height * 0.15), int(height * 0.85))

            ctrl_x = random.randint(0, width)
            ctrl_y = random.randint(0, height)

            color = tuple(np.random.randint(int(color_min), int(color_max) + 1, 3).tolist())
            thickness = random.randint(int(thick_min), int(thick_max))
            n_points = random.randint(int(pmin), int(pmax))
            points = []
            for i in range(n_points):
                t = i / max(1, (n_points - 1))
                x = int((1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * ctrl_x + t * t * x2)
                y = int((1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * ctrl_y + t * t * y2)
                points.append((x, y))

            for i in range(len(points) - 1):
                cv2.line(overlay, points[i], points[i + 1], color, thickness)

        alpha = random.uniform(float(alpha_min), float(alpha_max))
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image

    def add_noise(self, image, noise_type='gaussian'):
        """Add noise (low density)."""
        if noise_type == 'gaussian':
            mean = 0
            sigma = self._rand_float('noise.gaussian.sigma', 3.0, 12.0)
            gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
            noisy = image.astype(np.float32) + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        elif noise_type == 'salt_pepper':
            noisy = image.copy()
            salt_ratio = self._rand_float('noise.salt_pepper.salt_ratio', 0.003, 0.010)
            pepper_ratio = self._rand_float('noise.salt_pepper.pepper_ratio', 0.003, 0.010)
            h, w = image.shape[:2]
            num_salt = int(h * w * salt_ratio)
            num_pepper = int(h * w * pepper_ratio)

            ys = np.random.randint(0, h, num_salt)
            xs = np.random.randint(0, w, num_salt)
            noisy[ys, xs] = 255

            ys = np.random.randint(0, h, num_pepper)
            xs = np.random.randint(0, w, num_pepper)
            noisy[ys, xs] = 0

        elif noise_type == 'speckle':
            intensity = self._rand_float('noise.speckle.intensity', 0.02, 0.07)
            gauss = np.random.randn(*image.shape).astype(np.float32) * intensity
            noisy = image.astype(np.float32) + image.astype(np.float32) * gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        else:
            noisy = image.copy()

        jpeg_cfg = self._get_range('noise.jpeg_artifact', {})
        if isinstance(jpeg_cfg, dict) and jpeg_cfg.get('enabled', False):
            p = float(jpeg_cfg.get('p', 0.35))
            if random.random() < p:
                qmin, qmax = jpeg_cfg.get('quality', [55, 92])
                quality = random.randint(int(qmin), int(qmax))
                ok, enc = cv2.imencode('.jpg', noisy, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    noisy = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        return noisy

    def add_character_adhesion(self, image, char_positions):
        """Add partial character adhesion (overlapping but still distinguishable)."""
        prob = float(self._get_range('adhesion.probability', 0.22))
        alpha_range = self._get_range('adhesion.alpha', [0.18, 0.42])
        for i in range(len(char_positions) - 1):
            if random.random() < prob:
                x1, y1, w1, h1 = char_positions[i]
                x2, y2, w2, h2 = char_positions[i + 1]

                # Calculate overlap region (minimal overlap)
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)

                if overlap_x > 0 and overlap_y > 0:
                    # Create overlapping region
                    overlap_region = image[max(y1, y2):min(y1 + h1, y2 + h2),
                                          max(x1, x2):min(x1 + w1, x2 + w2)]

                    alpha = random.uniform(float(alpha_range[0]), float(alpha_range[1]))
                    blended = cv2.addWeighted(overlap_region, alpha, overlap_region, 1 - alpha, 0)

                    # Replace overlap region
                    image[max(y1, y2):min(y1 + h1, y2 + h2),
                          max(x1, x2):min(x1 + w1, x2 + w2)] = blended

        return image

    def _load_font_pool(self):
        project_root = get_config().get_project_root()
        fonts_dir = project_root / "fonts"
        font_files = []
        if fonts_dir.is_dir():
            font_files = sorted([str(p) for p in fonts_dir.glob("*.ttf")])
        if not font_files:
            fallback_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
                "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            ]
            for fp in fallback_paths:
                if os.path.exists(fp):
                    font_files = [fp]
                    break
        pool = []
        for fp in font_files:
            try:
                test_font = ImageFont.truetype(fp, 16)
                pool.append(fp)
            except Exception:
                continue
        return pool

    def _get_font(self, font_size, font_path=None):
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, int(font_size))
            except Exception:
                pass
        if self._font_pool:
            fp = random.choice(self._font_pool)
            try:
                return ImageFont.truetype(fp, int(font_size))
            except Exception:
                pass
        return ImageFont.load_default()

    def _render_character_layer(self, char, font_size, rotation_deg, scale, shear, font=None):
        if font is None:
            font = self._get_font(font_size)
        tmp = Image.new('RGBA', (font_size * 4, font_size * 4), (0, 0, 0, 0))
        d = ImageDraw.Draw(tmp)
        dark_colors = [(0, 0, 0, 255), (20, 20, 20, 255), (0, 40, 0, 255), (40, 0, 0, 255), (0, 0, 60, 255)]
        bbox = d.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        ox = (tmp.size[0] - tw) // 2
        oy = (tmp.size[1] - th) // 2
        
        stroke_width = random.randint(1, 4) if random.random() < 0.5 else 0
        d.text((ox, oy), char, font=font, fill=random.choice(dark_colors), stroke_width=stroke_width)

        if shear != 0:
            a = 1
            b = shear
            c = 0
            d0 = 0
            e = 1
            f = 0
            tmp = tmp.transform(tmp.size, Image.AFFINE, (a, b, c, d0, e, f), resample=Image.BICUBIC)

        tmp = tmp.rotate(rotation_deg, resample=Image.BICUBIC, expand=True)
        if scale != 1.0:
            nw = max(1, int(tmp.size[0] * scale))
            nh = max(1, int(tmp.size[1] * scale))
            tmp = tmp.resize((nw, nh), resample=Image.BICUBIC)

        tmp = self._apply_font_variation(tmp, font_size)

        bb = tmp.getbbox()
        if bb:
            tmp = tmp.crop(bb)
        return tmp

    def _apply_font_variation(self, char_img, font_size):
        """
        Randomly apply font variations for diversity.

        Variations include: stroke (bold), shear (italic), horizontal scale (narrow),
        sharpening/blurring, small-angle rotation, brightness/contrast changes.
        """
        from PIL import ImageEnhance

        if char_img.mode != 'RGBA':
            char_img = char_img.convert('RGBA')

        # Shear for italic effect
        if random.random() < 0.5:
            shear_factor = random.uniform(0.1, 0.35) * random.choice([-1, 1])
            a, b, c = 1, shear_factor, 0
            d0, e, f = 0, 1, 0
            char_img = char_img.transform(
                char_img.size,
                Image.AFFINE,
                (a, b, c, d0, e, f),
                resample=Image.BICUBIC
            )

        # Horizontal scale for narrow font
        if random.random() < 0.5:
            scale_factor = random.uniform(0.75, 0.90)
            w, h = char_img.size
            new_w = max(1, int(w * scale_factor))
            char_img = char_img.resize((new_w, h), resample=Image.BICUBIC)

        # Sharpening/blurring
        if random.random() < 0.5:
            if random.random() < 0.5:
                char_img = char_img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=0))
            else:
                radius = random.uniform(1, 2)
                char_img = char_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Small-angle rotation
        if random.random() < 0.5:
            if random.random() < 0.5:
                angle = random.uniform(-8, -2)
            else:
                angle = random.uniform(2, 8)
            char_img = char_img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Random brightness variation
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)
            enhancer = ImageEnhance.Brightness(char_img)
            char_img = enhancer.enhance(factor)

        # Random contrast variation
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(char_img)
            char_img = enhancer.enhance(factor)

        return char_img

    def draw_character(self, base_image, char, x, y, font_size, font=None):
        """Draw a single character with slight distortion, rotation, and scaling."""
        rotation_min, rotation_max = self._get_range('character.rotation', [-10, 10])
        scale_min, scale_max = self._get_range('character.scale', [0.90, 1.12])
        shear_min, shear_max = self._get_range('character.shear', [-0.12, 0.12])
        angle = random.randint(int(rotation_min), int(rotation_max))
        scale = random.uniform(float(scale_min), float(scale_max))
        shear = random.uniform(float(shear_min), float(shear_max))

        layer = self._render_character_layer(char, int(font_size), angle, scale, shear, font=font)
        if layer.mode != 'RGBA':
            layer = layer.convert('RGBA')
        base_image.paste(layer, (int(x), int(y)), layer)

        w, h = layer.size
        return (int(x), int(y), int(w), int(h))

    def _apply_wave_distortion(self, image):
        wave_cfg = self._get_range('distortion.wave', {})
        if not isinstance(wave_cfg, dict) or not wave_cfg.get('enabled', False):
            return image
        p = float(wave_cfg.get('p', 0.35))
        if random.random() >= p:
            return image

        amp_min, amp_max = wave_cfg.get('amplitude', [1.0, 2.5])
        per_min, per_max = wave_cfg.get('period', [24.0, 60.0])
        amplitude = random.uniform(float(amp_min), float(amp_max))
        period = random.uniform(float(per_min), float(per_max))

        h, w = image.shape[:2]
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            shift = amplitude * np.sin(2.0 * np.pi * y / period)
            map_x[y, :] = np.arange(w, dtype=np.float32) + shift
            map_y[y, :] = y
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def generate_single_captcha(self):
        """
        Generate a single CAPTCHA.

        Returns:
            image: CAPTCHA image (BGR)
            text: CAPTCHA label text
            char_positions: list of character positions
        """
        length = random.choice([4, 5, 6])
        text = ''.join(random.choice(self.characters) for _ in range(length))

        cv_image = self.generate_complex_background(self.image_width, self.image_height)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        char_width = self.image_width // (length + 1)
        char_height = int(self.image_height * random.uniform(0.6, 0.8))
        font_size = int(char_height * 0.9)

        captcha_font = self._get_font(font_size)

        char_positions = []
        offset_min, offset_max = self._get_range('character.offset', [-8, 8])
        overlap_min, overlap_max = self._get_range('character.overlap_ratio', [0.05, 0.22])
        step = max(1, int(char_width - char_width * random.uniform(overlap_min, overlap_max)))
        x = max(1, int(char_width * 0.25))
        for char in text:
            y = (self.image_height - char_height) // 2 + random.randint(int(offset_min), int(offset_max))
            x_jitter = random.randint(int(offset_min), int(offset_max))
            pos = self.draw_character(pil_image, char, x + x_jitter, y, font_size, font=captcha_font)
            char_positions.append(pos)
            x += step

        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Add character adhesion
        image = self.add_character_adhesion(image, char_positions)

        image = self._apply_wave_distortion(image)

        # Add interference lines
        image = self.add_interference_lines(image)

        # Add noise
        noise_type = random.choice(['gaussian', 'salt_pepper', 'speckle'])
        image = self.add_noise(image, noise_type)

        return image, text, char_positions

    def check_occlusion(self, image, char_positions, threshold=0.15):
        """Check character occlusion ratio."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for pos in char_positions:
            x, y, w, h = pos

            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                continue

            char_region = binary[y:y+h, x:x+w]

            total_pixels = char_region.size

            if total_pixels == 0:
                continue

            foreground_pixels = np.sum(char_region == 0)

            occlusion_ratio = foreground_pixels / total_pixels

            if threshold <= 0:
                return True

            if occlusion_ratio < (1 - threshold):
                return False

        return True

    def check_truncation(self, image, char_positions, margin=5):
        """Check if characters are truncated at image edges."""
        if margin <= 0:
            return False

        height, width = image.shape[:2]
        for pos in char_positions:
            x, y, w, h = pos

            if x < margin or y < margin or (x + w) > (width - margin) or (y + h) > (height - margin):
                return True

        return False

    def calculate_background_complexity(self, image):
        """Calculate background complexity metrics."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        std_dev = np.std(gray)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return {
            'entropy': entropy,
            'std_dev': std_dev,
            'edge_density': edge_density
        }

    def check_blur(self, image, threshold=60):
        """Check image blur using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return laplacian_var < threshold

    def check_contrast(self, image, threshold=30):
        """Check image contrast using standard deviation."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        std_dev = np.std(gray)

        return std_dev < threshold

    def check_quality(self, image, char_positions):
        """
        Comprehensive quality check.

        Returns:
            tuple: (passed, quality metrics dict)
        """
        bg_complexity = self.calculate_background_complexity(image)

        occlusion_ok = self.check_occlusion(image, char_positions, threshold=self._quality_config['occlusion_threshold'])
        truncation_ok = not self.check_truncation(image, char_positions, margin=self._quality_config['truncation_margin'])
        blur_ok = not self.check_blur(image, threshold=self._quality_config['blur_threshold'])
        contrast_ok = not self.check_contrast(image, threshold=self._quality_config['contrast_threshold'])

        # Background complexity threshold from config.yaml
        entropy_threshold = self._quality_config['entropy_threshold']
        # Disable check if threshold is <= 0 (background is inherently complex for CAPTCHAs)
        if entropy_threshold <= 0:
            bg_complexity_ok = True
        else:
            bg_complexity_ok = bg_complexity['entropy'] > entropy_threshold

        quality_ok = occlusion_ok and truncation_ok and blur_ok and contrast_ok and bg_complexity_ok

        quality_metrics = {
            'occlusion_ok': occlusion_ok,
            'truncation_ok': truncation_ok,
            'blur_ok': blur_ok,
            'contrast_ok': contrast_ok,
            'bg_complexity_ok': bg_complexity_ok,
            'bg_complexity': bg_complexity
        }

        return quality_ok, quality_metrics

    def save_captcha(self, image, text, filename):
        """Save CAPTCHA image and label file."""
        image_path = os.path.join(self.output_dir, f"{filename}.png")
        cv2.imwrite(image_path, image)

        label_path = os.path.join(self.output_dir, f"{filename}.txt")
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if self.verbose:
            logger.info(f"Image: {image_path}")
            logger.info(f"Label: {text}")

    def _generate_single_worker(self, args):
        """
        Multiprocessing worker: generate a single CAPTCHA.

        Args:
            args: (worker_id, num_samples) tuple
        """
        worker_id, num = args
        results = []

        for _ in range(num):
            try:
                image, text, char_positions = self.generate_single_captcha()
                quality_ok, quality_metrics = self.check_quality(image, char_positions)

                if quality_ok:
                    results.append((image, text, char_positions))
            except Exception as e:
                logger.exception(f"Worker {worker_id} error generating CAPTCHA: {e}")
                continue

        return results

    def _save_batch(self, batch):
        """Save a batch of CAPTCHAs."""
        for image, text, char_positions in batch:
            filename = f"{text}_{uuid.uuid4().hex[:8]}"
            self.save_captcha(image, text, filename)

    def generate_dataset(self):
        """Generate the full dataset with real-time quality checks."""
        logger.info(f"Starting CAPTCHA dataset generation...")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Number of samples: {self.num_samples}")
        logger.info(f"Character set: {self.characters}")
        logger.info(f"Character length: 4/5/6 random")
        logger.info(f"Quality check: disabled (quality check will be done during cleaning)")
        logger.info(f"Multiprocessing: {'enabled' if self.use_multiprocessing else 'disabled'}")

        generation_stats = {
            'total_generated': 0,
            'total_saved': 0,
        }

        if self.use_multiprocessing:
            # Multiprocessing generation
            num_workers = min(cpu_count(), 8)
            samples_per_worker = (self.num_samples // num_workers) + 10

            logger.info(f"Using {num_workers} processes for parallel generation...")

            with Pool(num_workers) as pool:
                tasks = [(i, samples_per_worker) for i in range(num_workers)]
                results = pool.map(self._generate_single_worker, tasks)

                all_captchas = []
                for worker_results in results:
                    all_captchas.extend(worker_results)

                all_captchas = all_captchas[:self.num_samples]

                logger.info(f"\nBatch saving {len(all_captchas)} CAPTCHAs...")
                batch_size = 500
                with tqdm(total=len(all_captchas), desc="Saving CAPTCHAs") as pbar:
                    for i in range(0, len(all_captchas), batch_size):
                        batch = all_captchas[i:i+batch_size]
                        self._save_batch(batch)
                        pbar.update(len(batch))

                generation_stats['total_generated'] = len(all_captchas)
                generation_stats['total_saved'] = len(all_captchas)
        else:
            # Single-process generation (original logic)
            with tqdm(total=self.num_samples, desc="Generating CAPTCHAs") as pbar:
                while generation_stats['total_saved'] < self.num_samples:
                    try:
                        image, text, char_positions = self.generate_single_captcha()
                        generation_stats['total_generated'] += 1

                        if generation_stats['total_generated'] % 10 == 0:
                            pbar.write(f"Generated: {generation_stats['total_generated']}, Saved: {generation_stats['total_saved']}")

                        # Skip quality check during generation (done later during cleaning)
                        filename = f"{text}_{uuid.uuid4().hex[:8]}"
                        self.save_captcha(image, text, filename)
                        generation_stats['total_saved'] += 1
                        pbar.update(1)

                    except Exception as e:
                        logger.exception(f"\nError generating CAPTCHA: {e}")
                        continue

        logger.info(f"\nCAPTCHA dataset generation complete!")
        logger.info(f"Total CAPTCHAs generated: {generation_stats['total_saved']}")
        logger.info(f"Saved to: {self.output_dir}")
        logger.info(f"Generation statistics:")
        logger.info(f"  Total generation attempts: {generation_stats['total_generated']}")
        logger.info(f"  Total saved: {generation_stats['total_saved']}")
        logger.info(f"Note: Quality check will be done during cleaning phase")
        logger.info(f"      No quality check during generation to avoid redundant work")
        return self.num_samples



def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CAPTCHA Dataset Generator')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (reads from config.yaml if not provided)')
    parser.add_argument('--multiprocessing', action='store_true', help='Enable multiprocessing (recommended for DSW environment)')
    args = parser.parse_args()

    config_loader = get_config()

    random.seed(42)
    np.random.seed(42)

    # Production mode: use --num_samples, save to data/raw_captchas/
    output_dir = config_loader.get_data_dir('raw')
    num_samples = int(args.num_samples) if args.num_samples is not None else int(config_loader.get('data.generate.num_samples', 20000))
    use_multiprocessing = args.multiprocessing
    logger.info("="*60)
    logger.info(f"Production mode: generating {num_samples} CAPTCHAs")
    logger.info("="*60)

    # Create generator
    config = get_config()
    original_width, original_height = config.get_original_image_size()

    generator = CaptchaGenerator(
        output_dir=output_dir,
        num_samples=num_samples,
        image_width=original_width,
        image_height=original_height,
        verbose=False,  # Production mode: skip per-image logging
        use_multiprocessing=use_multiprocessing
    )

    # Generate dataset
    generator.generate_dataset()


if __name__ == "__main__":
    main()
