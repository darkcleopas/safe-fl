import os
import yaml
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


class ModelUpdate(BaseModel):
    client_id: int
    weights: List[str]  # Base64 encoded weights
    num_examples: int
    local_loss: float
    local_accuracy: float

# Carregar configuração
config_path = os.environ.get('CONFIG_PATH', 'config/default.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

server_type = config.get('server', {}).get('type', 'standard')

try:
    # Importar classe do servidor
    server_module = __import__(f"src.servers.{server_type}_server", fromlist=['FLServer'])
    FLServer = server_module.FLServer
except ImportError:
    raise ImportError(f"Server type '{server_type}' not found")

# Criar aplicação FastAPI
app = FastAPI(title="Federated Learning Server")

# Instância do servidor FL
fl_server = None


@app.on_event("startup")
async def startup_event():
    global fl_server
    
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

@app.get("/client_status/{client_id}")
async def get_client_status(client_id: int):
    """Retorna se o cliente está ativo para treinamento."""
    if fl_server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    is_selected = client_id in fl_server.selected_clients
    is_active = client_id in fl_server.active_clients
    is_completed = client_id in fl_server.round_updates
    
    return {
        "round": fl_server.current_round,
        "is_selected": is_selected,
        "is_active": is_active,
        "is_completed": is_completed,
        "should_train": is_selected and is_active and not is_completed
    }


if __name__ == "__main__":
    # Obter endereço e porta do servidor
    host, port = config['server']['address'].split(":")

    # Executar servidor diretamente
    uvicorn.run("run_server:app", host=host, port=int(port), reload=False)
