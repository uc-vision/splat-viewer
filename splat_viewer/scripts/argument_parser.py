from argparse import ArgumentParser
from splat_viewer.viewer.settings import Settings, ViewMode


def add_settings_arguments(parser: ArgumentParser) -> None:
    """Add curated settings arguments that are useful for rendering."""
    defaults = Settings()
    
    # Rendering quality and appearance
    parser.add_argument('--view-mode', choices=['normal', 'depth', 'points'], 
                       default='normal', help='View mode (default: normal)')
    parser.add_argument('--point-size', type=float, default=defaults.point_size,
                       help=f'Point size for points view mode (default: {defaults.point_size})')
    
    # Animation settings
    parser.add_argument('--animate-speed', type=float, default=defaults.animate_speed,
                       help=f'Animation speed multiplier (default: {defaults.animate_speed})')
    parser.add_argument('--animate-pausing', type=float, default=defaults.animate_pausing,
                       help=f'Animation pausing/smoothing factor (default: {defaults.animate_pausing})')
    
    # Depth settings for depth view mode
    parser.add_argument('--depth-near', type=float, default=defaults.depth_near,
                       help=f'Near clipping plane for depth view (default: {defaults.depth_near})')
    parser.add_argument('--depth-far', type=float, default=defaults.depth_far,
                       help=f'Far clipping plane for depth view (default: {defaults.depth_far})')


def create_settings_from_args(args) -> Settings:
    """Create Settings instance from parsed arguments."""
    return Settings(
        view_mode=ViewMode[args.view_mode.title()],
        point_size=args.point_size,
        animate_speed=args.animate_speed,
        animate_pausing=args.animate_pausing,
        depth_near=args.depth_near,
        depth_far=args.depth_far
    )