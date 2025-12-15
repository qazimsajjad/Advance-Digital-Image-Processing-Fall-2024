"""
Region Splitting and Merging for Image Segmentation

This module implements the Region Split and Merge algorithm for image segmentation
using a quadtree data structure. The algorithm works in two phases:
1. Splitting: Recursively split non-homogeneous regions into quadrants
2. Merging: Merge adjacent similar regions to create larger uniform regions

References:
- Horwitz and Pavlidis quadtree-based segmentation
- Digital Image Processing by Gonzalez and Woods
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
import cv2


class Region:
    """Represents a rectangular region in the image with its properties."""
    
    def __init__(self, x: int, y: int, width: int, height: int, label: int = 0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.mean_value = None
        self.std_value = None
        self.min_value = None
        self.max_value = None
    
    def __repr__(self):
        return f"Region(x={self.x}, y={self.y}, w={self.width}, h={self.height}, label={self.label})"


class RegionSplitMerge:
    """
    Region Splitting and Merging algorithm for image segmentation.
    
    The algorithm uses a quadtree structure to recursively split the image
    into homogeneous regions and then merges similar adjacent regions.
    """
    
    def __init__(self, min_region_size: int = 4, std_threshold: float = 10.0, 
                 intensity_threshold: float = 15.0):
        """
        Initialize the Region Split and Merge segmentation algorithm.
        
        Parameters:
        -----------
        min_region_size : int
            Minimum size (width or height) of a region before stopping splitting
        std_threshold : float
            Standard deviation threshold for homogeneity test
        intensity_threshold : float
            Intensity difference threshold for homogeneity test (max - min)
        """
        self.min_region_size = min_region_size
        self.std_threshold = std_threshold
        self.intensity_threshold = intensity_threshold
        self.regions = []
        self.label_counter = 1
        
    def is_homogeneous(self, image: np.ndarray, region: Region) -> bool:
        """
        Check if a region is homogeneous based on standard deviation and intensity range.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image
        region : Region
            Region to test for homogeneity
            
        Returns:
        --------
        bool : True if region is homogeneous, False otherwise
        """
        # Extract region from image
        roi = image[region.y:region.y + region.height, 
                   region.x:region.x + region.width]
        
        if roi.size == 0:
            return True
        
        # Calculate statistics
        region.mean_value = np.mean(roi)
        region.std_value = np.std(roi)
        region.min_value = np.min(roi)
        region.max_value = np.max(roi)
        
        # Check homogeneity criteria
        intensity_range = region.max_value - region.min_value
        
        return (region.std_value <= self.std_threshold and 
                intensity_range <= self.intensity_threshold)
    
    def split(self, image: np.ndarray, region: Region) -> List[Region]:
        """
        Recursively split a region into four quadrants if it's not homogeneous.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image
        region : Region
            Region to potentially split
            
        Returns:
        --------
        List[Region] : List of resulting regions after splitting
        """
        # Base case: region is too small or homogeneous
        if (region.width <= self.min_region_size or 
            region.height <= self.min_region_size or
            self.is_homogeneous(image, region)):
            region.label = self.label_counter
            self.label_counter += 1
            return [region]
        
        # Split into four quadrants
        mid_x = region.width // 2
        mid_y = region.height // 2
        
        quadrants = [
            Region(region.x, region.y, mid_x, mid_y),  # Top-left
            Region(region.x + mid_x, region.y, region.width - mid_x, mid_y),  # Top-right
            Region(region.x, region.y + mid_y, mid_x, region.height - mid_y),  # Bottom-left
            Region(region.x + mid_x, region.y + mid_y, 
                  region.width - mid_x, region.height - mid_y)  # Bottom-right
        ]
        
        # Recursively split each quadrant
        result_regions = []
        for quad in quadrants:
            result_regions.extend(self.split(image, quad))
        
        return result_regions
    
    def are_adjacent(self, r1: Region, r2: Region) -> bool:
        """
        Check if two regions are adjacent (share a border).
        
        Parameters:
        -----------
        r1, r2 : Region
            Two regions to check for adjacency
            
        Returns:
        --------
        bool : True if regions are adjacent, False otherwise
        """
        # Check horizontal adjacency
        if (r1.y < r2.y + r2.height and r1.y + r1.height > r2.y):
            if (r1.x + r1.width == r2.x or r2.x + r2.width == r1.x):
                return True
        
        # Check vertical adjacency
        if (r1.x < r2.x + r2.width and r1.x + r1.width > r2.x):
            if (r1.y + r1.height == r2.y or r2.y + r2.height == r1.y):
                return True
        
        return False
    
    def can_merge(self, image: np.ndarray, r1: Region, r2: Region) -> bool:
        """
        Check if two adjacent regions can be merged based on similarity.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image
        r1, r2 : Region
            Two regions to check for merging possibility
            
        Returns:
        --------
        bool : True if regions can be merged, False otherwise
        """
        if not self.are_adjacent(r1, r2):
            return False
        
        # Get region statistics if not already calculated
        if r1.mean_value is None:
            roi1 = image[r1.y:r1.y + r1.height, r1.x:r1.x + r1.width]
            r1.mean_value = np.mean(roi1)
        
        if r2.mean_value is None:
            roi2 = image[r2.y:r2.y + r2.height, r2.x:r2.x + r2.width]
            r2.mean_value = np.mean(roi2)
        
        # Check if mean values are similar
        mean_diff = abs(r1.mean_value - r2.mean_value)
        
        return mean_diff <= self.intensity_threshold
    
    def merge(self, image: np.ndarray, regions: List[Region]) -> List[Region]:
        """
        Merge adjacent similar regions.
        
        Parameters:
        -----------
        image : np.ndarray
            Input grayscale image
        regions : List[Region]
            List of regions to merge
            
        Returns:
        --------
        List[Region] : List of regions after merging
        """
        merged = True
        iteration = 0
        max_iterations = 100  # Prevent infinite loops
        
        while merged and iteration < max_iterations:
            merged = False
            iteration += 1
            new_regions = regions.copy()
            
            i = 0
            while i < len(new_regions):
                j = i + 1
                while j < len(new_regions):
                    if self.can_merge(image, new_regions[i], new_regions[j]):
                        # Merge regions by assigning same label
                        new_regions[j].label = new_regions[i].label
                        merged = True
                    j += 1
                i += 1
            
            regions = new_regions
        
        return regions
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[Region]]:
        """
        Perform region split and merge segmentation on the input image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (grayscale or color). If color, will be converted to grayscale.
            
        Returns:
        --------
        Tuple[np.ndarray, List[Region]] : 
            - Segmented image with region labels
            - List of regions found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Initialize
        self.regions = []
        self.label_counter = 1
        height, width = gray.shape
        
        # Step 1: Splitting phase
        print("Starting splitting phase...")
        initial_region = Region(0, 0, width, height)
        split_regions = self.split(gray, initial_region)
        print(f"Splitting complete. Created {len(split_regions)} regions.")
        
        # Step 2: Merging phase
        print("Starting merging phase...")
        merged_regions = self.merge(gray, split_regions)
        print(f"Merging complete. Final regions: {len(set(r.label for r in merged_regions))}")
        
        # Create labeled output image
        segmented = np.zeros_like(gray, dtype=np.int32)
        for region in merged_regions:
            segmented[region.y:region.y + region.height,
                     region.x:region.x + region.width] = region.label
        
        self.regions = merged_regions
        return segmented, merged_regions
    
    def visualize_segments(self, segmented: np.ndarray, 
                          colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Create a colored visualization of the segmented image.
        
        Parameters:
        -----------
        segmented : np.ndarray
            Segmented image with region labels
        colormap : int
            OpenCV colormap to use for visualization
            
        Returns:
        --------
        np.ndarray : Colored visualization of segments
        """
        # Normalize to 0-255 range
        normalized = ((segmented - segmented.min()) * 255.0 / 
                     (segmented.max() - segmented.min() + 1e-7)).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        
        return colored
    
    def draw_boundaries(self, image: np.ndarray, 
                       segmented: np.ndarray, 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 1) -> np.ndarray:
        """
        Draw region boundaries on the original image.
        
        Parameters:
        -----------
        image : np.ndarray
            Original image
        segmented : np.ndarray
            Segmented image with region labels
        color : Tuple[int, int, int]
            BGR color for boundaries
        thickness : int
            Thickness of boundary lines
            
        Returns:
        --------
        np.ndarray : Image with boundaries drawn
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
        
        # Draw boundaries for each region
        for region in self.regions:
            x, y, w, h = region.x, region.y, region.width, region.height
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
        
        return output


# Convenience functions for easy importing
def segment_image(image: np.ndarray, 
                 min_region_size: int = 4,
                 std_threshold: float = 10.0,
                 intensity_threshold: float = 15.0) -> Tuple[np.ndarray, List[Region]]:
    """
    Convenience function to segment an image using Region Split and Merge.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    min_region_size : int
        Minimum size of regions
    std_threshold : float
        Standard deviation threshold for homogeneity
    intensity_threshold : float
        Intensity difference threshold
        
    Returns:
    --------
    Tuple[np.ndarray, List[Region]] : Segmented image and list of regions
    """
    segmenter = RegionSplitMerge(min_region_size, std_threshold, intensity_threshold)
    return segmenter.segment(image)


def segment_and_visualize(image: np.ndarray,
                         min_region_size: int = 4,
                         std_threshold: float = 10.0,
                         intensity_threshold: float = 15.0,
                         show_boundaries: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment an image and create visualizations.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    min_region_size : int
        Minimum size of regions
    std_threshold : float
        Standard deviation threshold
    intensity_threshold : float
        Intensity difference threshold
    show_boundaries : bool
        Whether to draw region boundaries
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray] : 
        - Segmented label image
        - Colored visualization
        - Image with boundaries (if show_boundaries=True)
    """
    segmenter = RegionSplitMerge(min_region_size, std_threshold, intensity_threshold)
    segmented, regions = segmenter.segment(image)
    
    # Create visualizations
    colored = segmenter.visualize_segments(segmented)
    
    if show_boundaries:
        boundaries = segmenter.draw_boundaries(image, segmented)
        return segmented, colored, boundaries
    
    return segmented, colored, None


if __name__ == "__main__":
    # Example usage
    print("Region Splitting and Merging Segmentation Module")
    print("=" * 50)
    print("\nUsage example:")
    print("from region_split_merge import segment_image, RegionSplitMerge")
    print("\n# Simple segmentation:")
    print("segmented, regions = segment_image(image)")
    print("\n# With custom parameters:")
    print("segmenter = RegionSplitMerge(min_region_size=8, std_threshold=15)")
    print("segmented, regions = segmenter.segment(image)")
    print("colored = segmenter.visualize_segments(segmented)")
    print("\nModule loaded successfully!")