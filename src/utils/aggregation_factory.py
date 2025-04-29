from typing import Dict, Any

import src.strategies.aggregation_strategies as aggregation_strategies


class AggregationFactory:
    """
    Fábrica para criar estratégias de agregação com base no tipo.
    """
    
    @staticmethod
    def create_aggregation_strategy(strategy_type: str, config: Dict[str, Any] = None) -> aggregation_strategies.AggregationStrategy:
        """
        Cria e retorna uma estratégia de agregação com base no tipo.
        
        Args:
            strategy_type: Tipo de estratégia de agregação
            config: Configuração para a estratégia
            
        Returns:
            Uma instância de AggregationStrategy
            
        Raises:
            ValueError: Se o tipo de estratégia não for reconhecido
        """
        if strategy_type == "FED_AVG":
            return aggregation_strategies.FedAvgStrategy(config)
        elif strategy_type == "FED_MEDIAN":
            return aggregation_strategies.FedMedianStrategy(config)
        elif strategy_type == "TRIMMED_MEAN":
            return aggregation_strategies.TrimmedMeanStrategy(config)
        elif strategy_type == "KRUM":
            return aggregation_strategies.KrumStrategy(config)
        elif strategy_type == "MULTI_KRUM":
            config = config or {}
            config['multi_krum'] = True
            return aggregation_strategies.KrumStrategy(config)
        elif strategy_type == "CLUSTERING":
            return aggregation_strategies.ClusteringStrategy(config)
        else:
            raise ValueError(f"Estratégia de agregação não reconhecida: {strategy_type}")