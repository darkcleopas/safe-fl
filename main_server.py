import os
import yaml
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

from src.servers.standard_server import FLServer


class ModelUpdate(BaseModel):
    client_id: int
    weights: List[str]  # Base64 encoded weights
    num_examples: int
    local_loss: float
    local_accuracy: float

# Criar aplicação FastAPI
app = FastAPI(title="Federated Learning Server")

# Instância do servidor FL
fl_server = None

@app.on_event("startup")
async def startup_event():
    global fl_server
    
    # Carregar configuração
    config_path = os.environ.get('CONFIG_PATH', 'config/default.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Inicializar servidor
    fl_server = FLServer(config)
    
    # Iniciar primeira rodada
    fl_server.start_round()

@app.get("/")
async def root():
    return {"message": "Federated Learning Server"}

@app.get("/model")
async def get_model():
    """Retorna o modelo global atual."""
    if fl_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    return {
        "weights": fl_server.serialize_model_weights(),
        "round": fl_server.current_round
    }

@app.post("/update")
async def update_model(update: ModelUpdate):
    """Recebe atualização de um cliente."""
    if fl_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    result = fl_server.submit_update(
        update.client_id, 
        update.weights, 
        update.num_examples,
        update.local_loss,
        update.local_accuracy
    )
    
    return result

@app.get("/round")
async def get_round():
    """Retorna informações sobre a rodada atual."""
    if fl_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    status = fl_server.get_round_status()
    
    # Se a rodada atual está completa e não é a última, iniciar próxima rodada
    if status['is_round_complete'] and status['round'] < status['total_rounds']:
        fl_server.start_round()
        status = fl_server.get_round_status()
    
    return status

@app.get("/status")
async def get_status():
    """Retorna status geral do servidor."""
    if fl_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    return {
        "current_round": fl_server.current_round,
        "total_rounds": fl_server.total_rounds,
        "training_complete": fl_server.current_round >= fl_server.total_rounds,
        "metrics": {
            "rounds": fl_server.metrics['rounds'],
            "accuracy": fl_server.metrics['accuracy'],
            "loss": fl_server.metrics['loss']
        }
    }

if __name__ == "__main__":
    # Executar servidor diretamente
    uvicorn.run("main_server:app", host="0.0.0.0", port=8000, reload=False)