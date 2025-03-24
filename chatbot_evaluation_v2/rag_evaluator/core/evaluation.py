    """Manager for the evaluation process."""
    
    def __init__(
        self,
        config_manager: ConfigManager = None,
        data_manager: DataManager = None,
        metric_registry: MetricRegistry = None
    ):
        """Initialize evaluation manager.
        
        Args:
            config_manager: Configuration manager.
            data_manager: Data manager.
            metric_registry: Metric registry.
        """
        self.config_manager = config_manager or ConfigManager()
        self.data_manager = data_manager or DataManager()
        self.metric_registry = metric_registry or MetricRegistry()
        self.logger = logging.getLogger(__name__)
    
    def run_evaluation(
        self,
        config_name: str,
        dataset_name: str,
        output_dir: str = None