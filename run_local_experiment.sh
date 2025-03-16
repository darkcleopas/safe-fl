#!/bin/bash

# ===========================================
# Federated Learning Runner Script (Scalable)
# ===========================================

# Define colors for better visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --delay)
      DELAY_BETWEEN="$2"
      shift 2
      ;;
    --server-wait)
      SERVER_STARTUP_TIME="$2"
      shift 2
      ;;
    --port)
      SERVER_PORT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --config PATH      Path to config file (default: config/default.yaml)"
      echo "  --delay N          Delay between client startups in seconds (default: 10)"
      echo "  --server-wait N    Time to wait for server initialization in seconds (default: 15)"
      echo "  --port N           Server port (override config value)"
      echo "  --help             Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$DIR/.venv"
PARSER_SCRIPT="$DIR/parse_yaml_config.py"

# Verify that the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}Please create a virtual environment first.${NC}"
    exit 1
fi

# Verify that the config file exists
if [ ! -f "$DIR/$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config file not found at $CONFIG_PATH${NC}"
    exit 1
fi

# Create parser script if it doesn't exist
if [ ! -f "$PARSER_SCRIPT" ]; then
    echo -e "${YELLOW}Creating YAML parser script...${NC}"
    cp "$DIR/parse_yaml_config.py" "$PARSER_SCRIPT"
    chmod +x "$PARSER_SCRIPT"
fi

# Extract configuration using the Python parser
echo -e "${BLUE}Extraindo configuração de $CONFIG_PATH...${NC}"
CONFIG_JSON=$(python "$PARSER_SCRIPT" "$DIR/$CONFIG_PATH")
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro: Falha ao analisar o arquivo de configuração${NC}"
    exit 1
fi

# Verificar se o resultado é um JSON válido
if ! echo "$CONFIG_JSON" | python -c "import sys, json; json.load(sys.stdin)" &> /dev/null; then
    echo -e "${RED}Erro: A saída do parser não é um JSON válido:${NC}"
    echo "$CONFIG_JSON"
    exit 1
fi

# Extract values from the JSON output
NUM_CLIENTS=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['num_clients'])")
CONFIG_SERVER_HOST=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['server_host'])")
CONFIG_SERVER_PORT=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['server_port'])")
EXPERIMENT_NAME=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['experiment_name'])")
TOTAL_ROUNDS=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['rounds'])")
SEED=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['seed'])")
DATASET=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['dataset'])")
MODEL=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['model'])")
HONEST_CLIENT_TYPE=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['honest_client_type'])")
MALICIOUS_PERCENTAGE=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['malicious_percentage'])")
MALICIOUS_CLIENT_TYPE=$(echo "$CONFIG_JSON" | python -c "import sys, json; print(json.load(sys.stdin)['malicious_client_type'])")

# Calcular o número de clientes maliciosos
NUM_MALICIOUS=$(python -c "import math; print(math.floor($NUM_CLIENTS * $MALICIOUS_PERCENTAGE))")
NUM_HONEST=$(($NUM_CLIENTS - $NUM_MALICIOUS))

# Gerar lista de clientes maliciosos aleatoriamente
if [ $NUM_MALICIOUS -gt 0 ]; then
    MALICIOUS_CLIENTS=$(python -c "import random; random.seed($SEED); print(','.join(map(str, sorted(random.sample(range(1, $NUM_CLIENTS + 1), $NUM_MALICIOUS)))))")
    echo -e "${YELLOW}Clientes maliciosos selecionados: $MALICIOUS_CLIENTS${NC}"
else
    MALICIOUS_CLIENTS=""
fi

# Detect terminal emulator for Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v gnome-terminal &> /dev/null; then
        TERMINAL_TYPE="gnome"
    elif command -v xfce4-terminal &> /dev/null; then
        TERMINAL_TYPE="xfce4"
    elif command -v konsole &> /dev/null; then
        TERMINAL_TYPE="konsole"
    elif command -v xterm &> /dev/null; then
        TERMINAL_TYPE="xterm"
    else
        echo -e "${RED}Error: Could not find a supported terminal emulator.${NC}"
        echo -e "${YELLOW}Please install gnome-terminal, xfce4-terminal, konsole, or xterm.${NC}"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    TERMINAL_TYPE="macos"
else
    echo -e "${RED}Unsupported OS. Please run the commands manually.${NC}"
    exit 1
fi

# Configuração de ambiente básica
SERVER_URL="http://$CONFIG_SERVER_HOST:$CONFIG_SERVER_PORT"
DELAY_BETWEEN=10         # Default delay between clients (seconds)
SERVER_STARTUP_TIME=15   # Time to wait for server initialization (seconds)
CUDA_VISIBLE_DEVICES="-1"
TF_NUM_THREADS="4"
TF_CPP_MIN_LOG_LEVEL="1"

# Function to open a terminal and run a command
run_terminal() {
    local title="$1"
    local command="$2"
    
    case $TERMINAL_TYPE in
        "gnome")
            gnome-terminal --tab --title="$title" -- bash -c "$command"
            ;;
        "xfce4")
            xfce4-terminal --title="$title" --command="$command"
            ;;
        "konsole")
            konsole --new-tab --p tabtitle="$title" -e bash -c "$command"
            ;;
        "xterm")
            xterm -T "$title" -e bash -c "$command"
            ;;
        "macos")
            osascript -e "tell application \"Terminal\"" \
                      -e "do script \"$command\"" \
                      -e "set custom title of front window to \"$title\"" \
                      -e "end tell"
            ;;
    esac
}

# Function to run a server or client in a new terminal
run_component() {
    local title="$1"
    local component_type="$2"
    local client_id="$3"
    
    # Construct the environment variable exports
    local env_vars=""
    
    if [ "$component_type" = "server" ]; then
        env_vars="export CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" && "
        env_vars+="export CONFIG_PATH=\"$CONFIG_PATH\""
        command="cd \"$DIR\" && source \"$VENV_PATH/bin/activate\" && $env_vars && python run_server.py; exec bash"
    else
        # Determinar se este cliente é malicioso
        is_malicious=false
        client_type="$HONEST_CLIENT_TYPE"
        
        if [ ! -z "$MALICIOUS_CLIENTS" ]; then
            if [[ ",$MALICIOUS_CLIENTS," == *",$client_id,"* ]]; then
                is_malicious=true
                client_type="$MALICIOUS_CLIENT_TYPE"
                title="$title (Malicioso: $MALICIOUS_CLIENT_TYPE)"
            fi
        fi
        
        env_vars="export CLIENT_ID=\"$client_id\" && "
        env_vars+="export SERVER_URL=\"$SERVER_URL\" && "
        env_vars+="export CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" && "
        env_vars+="export CONFIG_PATH=\"$CONFIG_PATH\" && "
        env_vars+="export CLIENT_TYPE=\"$client_type\" && "
        env_vars+="export TF_NUM_THREADS=\"$TF_NUM_THREADS\" && "
        env_vars+="export TF_CPP_MIN_LOG_LEVEL=\"$TF_CPP_MIN_LOG_LEVEL\""
        
        if [ "$is_malicious" = true ]; then
            env_vars+=" && export IS_MALICIOUS=\"true\""
        fi
        
        command="cd \"$DIR\" && source \"$VENV_PATH/bin/activate\" && $env_vars && python run_client.py; exec bash"
    fi
    
    run_terminal "$title" "$command"
}

# Clear the screen and show banner
clear
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   Starting Federated Learning Environment      ${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Experiment: ${YELLOW}$EXPERIMENT_NAME${NC} Seed: ${YELLOW}$SEED${NC}"
echo -e "  Dataset: ${YELLOW}$DATASET${NC}"
echo -e "  Model: ${YELLOW}$MODEL${NC}"
echo -e "  Number of clients: ${YELLOW}$NUM_CLIENTS${NC} (from config)"
echo -e "  Total rounds: ${YELLOW}$TOTAL_ROUNDS${NC}"
echo -e "  Delay between clients: ${YELLOW}$DELAY_BETWEEN seconds${NC}"
echo -e "  Config file: ${YELLOW}$CONFIG_PATH${NC}"
echo -e "  Server host: ${YELLOW}$CONFIG_SERVER_HOST${NC}"
echo -e "  Server port: ${YELLOW}$CONFIG_SERVER_PORT${NC}"
echo -e "  Terminal type: ${YELLOW}$TERMINAL_TYPE${NC}"

# Informações sobre clientes maliciosos
if [ $NUM_MALICIOUS -gt 0 ]; then
    echo -e "${RED}Configuração de ataque:${NC}"
    echo -e "  Tipo de cliente honesto: ${YELLOW}$HONEST_CLIENT_TYPE${NC}"
    echo -e "  Porcentagem de clientes maliciosos: ${YELLOW}$(echo "$MALICIOUS_PERCENTAGE * 100" | bc)%${NC}"
    echo -e "  Número de clientes maliciosos: ${YELLOW}$NUM_MALICIOUS${NC} de $NUM_CLIENTS"
    echo -e "  Tipo de ataque: ${YELLOW}$MALICIOUS_CLIENT_TYPE${NC}"
    echo -e "  IDs dos clientes maliciosos: ${YELLOW}$MALICIOUS_CLIENTS${NC}"
else
    echo -e "${GREEN}Sem clientes maliciosos${NC}"
    echo -e "  Tipo de cliente: ${YELLOW}$HONEST_CLIENT_TYPE${NC}"
fi
echo ""

# Start the server
echo -e "${BLUE}Starting server on $CONFIG_SERVER_HOST:$CONFIG_SERVER_PORT...${NC}"
run_component "FL Server" "server"

# Give the server time to start
echo -e "${YELLOW}Waiting $SERVER_STARTUP_TIME seconds for server to initialize...${NC}"
for ((i=1; i<=$SERVER_STARTUP_TIME; i++)); do
    echo -ne "${YELLOW}[$i/$SERVER_STARTUP_TIME]${NC}\r"
    sleep 1
done
echo -e "${GREEN}Server initialization period complete.${NC}"

# Start each client with the appropriate delay
for ((i=1; i<=$NUM_CLIENTS; i++)); do
    echo -e "${BLUE}Starting Client $i...${NC}"
    run_component "FL Client $i" "client" "$i"
    
    # Apply delay if this isn't the last client
    if [ $i -lt $NUM_CLIENTS ]; then
        echo -e "${YELLOW}Waiting $DELAY_BETWEEN seconds before starting next client...${NC}"
        for ((j=1; j<=$DELAY_BETWEEN; j++)); do
            echo -ne "${YELLOW}[$j/$DELAY_BETWEEN]${NC}\r"
            sleep 1
        done
        echo -e "${GREEN}Starting next client.${NC}"
    fi
done

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   Federated Learning Environment Started       ${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Server and all $NUM_CLIENTS clients are now running in separate terminals."
echo -e "To stop the experiment, close each terminal or press CTRL+C in each window."
echo ""
echo -e "${YELLOW}Note: You can run this script with different options:${NC}"
echo -e "  ${BLUE}$0 --config configs/my_custom_config.yaml${NC}"
echo -e "  ${BLUE}$0 --help${NC} (for all options)"