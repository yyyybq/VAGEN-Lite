#!/usr/bin/env python3
"""
Test script for enhanced camera validation features.

Tests the three new validation functions:
1. count_visible_corners - Count visible AABB corners
2. calculate_projected_area_ratio - Calculate object's projected area
3. calculate_occlusion_area_2d - 2D image-space occlusion calculation
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from camera_sampler import (
    CameraSampler,
    count_visible_corners,
    calculate_projected_area_ratio,
    calculate_occlusion_area_2d,
    AABB,
)
from config import CameraSamplingConfig


def test_visible_corners():
    """Test visible corner counting."""
    print("\n" + "="*60)
    print("TEST 1: Visible Corner Counting")
    print("="*60)
    
    config = CameraSamplingConfig()
    sampler = CameraSampler(config)
    
    # Create a simple AABB (1m cube at origin)
    target_bmin = np.array([0.0, 0.0, 0.0])
    target_bmax = np.array([1.0, 1.0, 1.0])
    
    # Camera looking at cube from distance
    cam_pos = np.array([2.0, 2.0, 1.5])
    cam_target = np.array([0.5, 0.5, 0.5])  # Center of cube
    
    # Count visible corners
    visible_count = count_visible_corners(
        sampler.intrinsics, cam_pos, cam_target,
        target_bmin, target_bmax,
        config.image_width, config.image_height,
        border=2
    )
    
    print(f"Camera position: {cam_pos}")
    print(f"Target center: {cam_target}")
    print(f"Visible corners: {visible_count}/8")
    
    # Test with different camera positions
    test_positions = [
        (np.array([3.0, 0.5, 0.5]), "Side view"),
        (np.array([0.5, 3.0, 0.5]), "Front view"),
        (np.array([0.5, 0.5, 3.0]), "Top view"),
        (np.array([-1.0, -1.0, 0.5]), "Behind and below"),
    ]
    
    for pos, desc in test_positions:
        count = count_visible_corners(
            sampler.intrinsics, pos, cam_target,
            target_bmin, target_bmax,
            config.image_width, config.image_height
        )
        print(f"  {desc:20s}: {count}/8 corners visible")
    
    print("✓ Corner counting test completed")


def test_projected_area():
    """Test projected area calculation."""
    print("\n" + "="*60)
    print("TEST 2: Projected Area Ratio")
    print("="*60)
    
    config = CameraSamplingConfig()
    sampler = CameraSampler(config)
    
    # Create AABB
    target_bmin = np.array([0.0, 0.0, 0.0])
    target_bmax = np.array([1.0, 1.0, 1.0])
    cam_target = np.array([0.5, 0.5, 0.5])
    
    # Test at different distances
    test_distances = [
        (1.5, "Very close"),
        (3.0, "Medium distance"),
        (5.0, "Far"),
        (10.0, "Very far"),
    ]
    
    for dist, desc in test_distances:
        cam_pos = np.array([dist, 0.5, 0.5])
        area_ratio, projected_px = calculate_projected_area_ratio(
            sampler.intrinsics, cam_pos, cam_target,
            target_bmin, target_bmax,
            config.image_width, config.image_height
        )
        
        total_px = config.image_width * config.image_height
        print(f"  {desc:20s} (d={dist:4.1f}m): "
              f"ratio={area_ratio:6.2%}, pixels={projected_px:8.1f}/{total_px} "
              f"{'✓ PASS' if area_ratio >= 0.05 else '✗ FAIL (too small)'}")
    
    print("✓ Projected area test completed")


def test_2d_occlusion():
    """Test 2D image-space occlusion calculation."""
    print("\n" + "="*60)
    print("TEST 3: 2D Image-Space Occlusion")
    print("="*60)
    
    try:
        import cv2
        print("✓ OpenCV available - 2D occlusion enabled")
    except ImportError:
        print("✗ OpenCV not available - will use fallback 3D occlusion")
        return
    
    config = CameraSamplingConfig()
    sampler = CameraSampler(config)
    
    # Create target object
    target_bmin = np.array([0.0, 0.0, 0.0])
    target_bmax = np.array([1.0, 1.0, 1.0])
    
    # Create an occluder in front of target
    occluder = AABB(
        id="occluder_1",
        label="table",
        bmin=np.array([-0.5, 1.0, 0.0]),  # In front
        bmax=np.array([1.5, 1.5, 1.5])
    )
    
    cam_pos = np.array([0.5, 3.0, 0.5])
    cam_target = np.array([0.5, 0.5, 0.5])
    
    # Calculate occlusion with occluder
    occ_info = calculate_occlusion_area_2d(
        sampler.intrinsics, cam_pos, cam_target,
        target_bmin, target_bmax,
        [occluder],
        config.image_width, config.image_height,
        target_id="target"
    )
    
    print(f"Camera position: {cam_pos}")
    print(f"Target AABB: {target_bmin} to {target_bmax}")
    print(f"Occluder AABB: {occluder.bmin} to {occluder.bmax}")
    print(f"\nOcclusion results:")
    print(f"  Target area:    {occ_info['target_area_px']:8.1f} px")
    print(f"  Visible area:   {occ_info['visible_area_px']:8.1f} px")
    print(f"  Occluded area:  {occ_info['occluded_area_px']:8.1f} px")
    print(f"  Occlusion ratio: {occ_info['occlusion_ratio_target']:6.2%}")
    
    if occ_info['occlusion_ratio_target'] <= 0.7:
        print("  ✓ PASS - Occlusion acceptable (≤70%)")
    else:
        print("  ✗ FAIL - Too occluded (>70%)")
    
    # Test without occluder
    occ_info_clear = calculate_occlusion_area_2d(
        sampler.intrinsics, cam_pos, cam_target,
        target_bmin, target_bmax,
        [],  # No occluders
        config.image_width, config.image_height,
        target_id="target"
    )
    
    print(f"\nClear view (no occluders):")
    print(f"  Occlusion ratio: {occ_info_clear['occlusion_ratio_target']:6.2%}")
    print("✓ 2D occlusion test completed")


def test_integration():
    """Test integration with CameraSampler class methods."""
    print("\n" + "="*60)
    print("TEST 4: Integration with CameraSampler")
    print("="*60)
    
    config = CameraSamplingConfig()
    sampler = CameraSampler(config)
    
    # Setup test scenario
    target_bmin = np.array([0.0, 0.0, 0.0])
    target_bmax = np.array([1.0, 1.0, 1.0])
    cam_pos = np.array([2.0, 2.0, 1.5])
    cam_target = np.array([0.5, 0.5, 0.5])
    
    # Test 1: Check visible corners
    has_enough, count = sampler.check_visible_corners_count(
        cam_pos, cam_target, target_bmin, target_bmax, min_corners=1
    )
    print(f"1. Visible corners check: {count}/8 corners, min=1 → {'✓ PASS' if has_enough else '✗ FAIL'}")
    
    # Test 2: Check projected area
    is_large, ratio, pixels = sampler.check_projected_area(
        cam_pos, cam_target, target_bmin, target_bmax, min_area_ratio=0.05
    )
    print(f"2. Projected area check: {ratio:.2%} of image, min=5% → {'✓ PASS' if is_large else '✗ FAIL'}")
    
    # Test 3: Check 2D occlusion (if cv2 available)
    try:
        import cv2
        occluders = [
            AABB("occ1", "chair", 
                 np.array([-1.0, -1.0, 0.0]), 
                 np.array([-0.5, -0.5, 1.0]))
        ]
        is_acceptable, occ_info = sampler.check_occlusion_2d(
            cam_pos, cam_target, target_bmin, target_bmax,
            occluders, target_id="target", max_occlusion_ratio=0.7
        )
        print(f"3. 2D occlusion check: {occ_info['occlusion_ratio_target']:.2%} occluded, max=70% → "
              f"{'✓ PASS' if is_acceptable else '✗ FAIL'}")
    except ImportError:
        print("3. 2D occlusion check: ⊘ SKIPPED (cv2 not available)")
    
    print("✓ Integration test completed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Enhanced Camera Validation Tests")
    print("Testing 3 new features from ViewSuite integration")
    print("="*60)
    
    try:
        test_visible_corners()
        test_projected_area()
        test_2d_occlusion()
        test_integration()
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
