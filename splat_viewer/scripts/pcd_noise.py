import numpy as np
import open3d as o3d
from typing import Tuple, Optional

class PointCloudNoise:
    def __init__(self):
        pass

    def add_gaussian_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        mean: float = 0, 
        std: float = 0.01
    ) -> o3d.geometry.PointCloud:
        """Add Gaussian noise to the point cloud."""
        points = np.asarray(pcd.points)
        noise = np.random.normal(mean, std, points.shape)
        return self._create_noisy_pcd(pcd, points + noise)

    def add_uniform_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        low: float = -0.01, 
        high: float = 0.01
    ) -> o3d.geometry.PointCloud:
        """Add uniform noise to the point cloud."""
        points = np.asarray(pcd.points)
        noise = np.random.uniform(low, high, points.shape)
        return self._create_noisy_pcd(pcd, points + noise)

    def add_outliers(
        self,
        pcd: o3d.geometry.PointCloud,
        outlier_ratio: float = 0.05,  # 5% outliers by default
        outlier_scale: float = 2.0,
        method: str = "bounding_box",
        noise_std: float = 0.5,
        min_outliers: int = 1,  # Ensure at least 1 outlier
        max_outliers: Optional[int] = None,  # Optional cap
    ) -> o3d.geometry.PointCloud:
        """
        Add outliers as a percentage of the original point cloud size.
        
        Args:
            pcd: Input point cloud.
            outlier_ratio: Fraction of the original points to add as outliers (e.g., 0.05 = 5%).
            outlier_scale: Scale factor for outlier generation (relative to cloud size).
            method: 
                - "bounding_box": Outliers near the bounding box edges.
                - "gaussian": Outliers sampled from a scaled Gaussian distribution.
            noise_std: Standard deviation for jitter (if using "bounding_box").
            min_outliers: Minimum number of outliers (even if ratio is tiny).
            max_outliers: Optional maximum number of outliers.
        
        Returns:
            Noisy point cloud with outliers.
        """
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return pcd

        # Calculate number of outliers
        n_outliers = int(len(points) * outlier_ratio)
        n_outliers = max(min_outliers, n_outliers)  # Enforce minimum
        if max_outliers is not None:
            n_outliers = min(max_outliers, n_outliers)  # Enforce maximum

        # Compute cloud statistics
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2
        cloud_size = max_bound - min_bound

        # Generate outliers
        if method == "bounding_box":
            outliers = np.random.rand(n_outliers, 3) * (cloud_size * outlier_scale)
            outliers += center - (cloud_size * outlier_scale) / 2
            outliers += np.random.normal(0, noise_std, (n_outliers, 3))  # Jitter
        elif method == "gaussian":
            outliers = np.random.normal(center, cloud_size * outlier_scale, (n_outliers, 3))
        else:
            raise ValueError("Method must be 'bounding_box' or 'gaussian'")

        # Combine with original points
        noisy_points = np.vstack([points, outliers])
        noisy_pcd = self._create_noisy_pcd(pcd, noisy_points)

        # Handle colors
        if pcd.has_colors():
            outlier_colors = np.random.rand(n_outliers, 3)
            original_colors = np.asarray(pcd.colors)
            noisy_pcd.colors = o3d.utility.Vector3dVector(np.vstack([original_colors, outlier_colors]))

        return noisy_pcd

    def add_salt_pepper_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        prob: float = 0.01, 
        strength: float = 0.1
    ) -> o3d.geometry.PointCloud:
        """Add salt-and-pepper noise to the point cloud."""
        points = np.asarray(pcd.points)
        mask = np.random.rand(len(points)) < prob
        noise = np.random.choice([-strength, strength], size=(np.sum(mask), 3))
        points[mask] += noise
        return self._create_noisy_pcd(pcd, points)

    def add_bias_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        bias: Tuple[float, float, float] = (0.05, -0.02, 0.03)
    ) -> o3d.geometry.PointCloud:
        """Add a constant bias to all points."""
        points = np.asarray(pcd.points)
        points += np.array(bias)
        return self._create_noisy_pcd(pcd, points)

    def add_density_variation(
        self, 
        pcd: o3d.geometry.PointCloud, 
        keep_ratio: float = 0.8
    ) -> o3d.geometry.PointCloud:
        """Randomly downsample the point cloud to vary density."""
        return pcd.random_down_sample(keep_ratio)

    def add_structured_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        std_x: float = 0.01, 
        std_y: float = 0.01, 
        std_z: float = 0.05
    ) -> o3d.geometry.PointCloud:
        """Add axis-dependent Gaussian noise."""
        points = np.asarray(pcd.points)
        noise = np.random.normal([0, 0, 0], [std_x, std_y, std_z], points.shape)
        return self._create_noisy_pcd(pcd, points + noise)

    def add_temporal_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        time_scale: float = 0.02
    ) -> o3d.geometry.PointCloud:
        """Simulate motion artifacts by scaling noise with point index."""
        points = np.asarray(pcd.points)
        motion_noise = np.random.normal(0, time_scale, points.shape) * np.arange(len(points))[:, None]
        return self._create_noisy_pcd(pcd, points + motion_noise)

    def add_non_rigid_deformation(
        self, 
        pcd: o3d.geometry.PointCloud, 
        amplitude: float = 0.02, 
        frequency: float = 10
    ) -> o3d.geometry.PointCloud:
        """Apply sinusoidal non-rigid deformation."""
        points = np.asarray(pcd.points)
        x, y, z = points.T
        points[:, 0] += amplitude * np.sin(frequency * y)  # Deform along X-axis based on Y
        return self._create_noisy_pcd(pcd, points)

    def add_quantization_noise(
        self, 
        pcd: o3d.geometry.PointCloud, 
        step_size: float = 0.01
    ) -> o3d.geometry.PointCloud:
        """Simulate depth quantization artifacts."""
        points = np.asarray(pcd.points)
        points = np.round(points / step_size) * step_size
        return self._create_noisy_pcd(pcd, points)

    def _create_noisy_pcd(
        self, 
        pcd: o3d.geometry.PointCloud, 
        noisy_points: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """Helper function to create a noisy point cloud while preserving attributes."""
        noisy_pcd = o3d.geometry.PointCloud()
        noisy_pcd.points = o3d.utility.Vector3dVector(noisy_points)
        if pcd.has_colors():
            noisy_pcd.colors = pcd.colors
        if pcd.has_normals():
            noisy_pcd.normals = pcd.normals
        return noisy_pcd