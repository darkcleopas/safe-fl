from typing import Dict, Any, List

# Importar as novas classes base e implementações
from src.strategies.abc_strategies import AggregationStrategy, ClientFilterStrategy, GlobalModelFilterStrategy
from src.strategies.aggregation_strategies import FedAvgStrategy, FedMedianStrategy, TrimmedMeanStrategy
from src.strategies.client_filter_strategies import (
    AdaptiveL2ClientFilter, KrumClientFilter,
    ClusteringClientFilter,
    AdaptiveClusteringClientFilter,
    DynamicLayerPCAFilter,
    PCAGeometricMedianDistanceFilter,
    PCAGeometricMedianDirectionFilter
)
from src.strategies.global_model_filter_strategies import AdaptiveL2GlobalModelFilter

class StrategyFactory:
    """Fábrica para criar todos os tipos de estratégias da pipeline de defesa."""

    AGGREGATION_MAP = {
        "FED_AVG": FedAvgStrategy,
        "FED_MEDIAN": FedMedianStrategy,
        "TRIMMED_MEAN": TrimmedMeanStrategy,
    }
    
    CLIENT_FILTER_MAP = {
        "L2_DIRECTIONAL_FILTER": AdaptiveL2ClientFilter,
        "KRUM": KrumClientFilter,
        "MULTI_KRUM": KrumClientFilter, # Usa a mesma classe, mas com config diferente
        "CLUSTERING": ClusteringClientFilter,
        "ADAPTIVE_CLUSTERING": AdaptiveClusteringClientFilter,
        "DYNAMIC_LAYER_PCA": DynamicLayerPCAFilter,
        "PCA_GEOMETRIC_MEDIAN_DISTANCE": PCAGeometricMedianDistanceFilter,
        "PCA_GEOMETRIC_MEDIAN_DIRECTION": PCAGeometricMedianDirectionFilter,
    }
    
    GLOBAL_MODEL_FILTER_MAP = {
        "L2_GLOBAL_MODEL_FILTER": AdaptiveL2GlobalModelFilter,
    }

    @staticmethod
    def create_aggregation_strategy(config: Dict[str, Any]) -> AggregationStrategy:
        strategy_name = config.get('name', 'FED_AVG').upper()
        strategy_class = StrategyFactory.AGGREGATION_MAP.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Estratégia de agregação não reconhecida: {strategy_name}")
        return strategy_class(config.get('params'))

    @staticmethod
    def create_client_filters(configs: List[Dict[str, Any]]) -> List[ClientFilterStrategy]:
        filters = []
        if not configs:
            return filters
        
        for config in configs:
            filter_name = config.get('name').upper()
            
            # Caso especial para MULTI_KRUM
            if filter_name == "MULTI_KRUM":
                params = config.get('params', {})
                params['multi_krum'] = True
            else:
                params = config.get('params')

            filter_class = StrategyFactory.CLIENT_FILTER_MAP.get(filter_name)
            if not filter_class:
                raise ValueError(f"Filtro de cliente não reconhecido: {filter_name}")
            filters.append(filter_class(params))
        return filters

    @staticmethod
    def create_global_model_filter(config: Dict[str, Any]) -> GlobalModelFilterStrategy:
        if not config:
            return None
            
        filter_name = config.get('name').upper()
        filter_class = StrategyFactory.GLOBAL_MODEL_FILTER_MAP.get(filter_name)
        if not filter_class:
            raise ValueError(f"Filtro de modelo global não reconhecido: {filter_name}")
        return filter_class(config.get('params'))