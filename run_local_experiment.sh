#!/bin/bash

# ===========================================
# Federated Learning Runner Script (Scalable)
# ===========================================

# Default configuration
NUM_CLIENTS=5            # Default number of clients
DELAY_BETWEEN=10         # Default delay between clients (seconds)
SERVER_STARTUP_TIME=15   # Time to wait for server initialization (seconds)
CONFIG_PATH="config/default.yaml"
SERVER_PORT=8000

# Define colors for better visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clients)
      NUM_CLIENTS="$2"
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
    --config)
      CONFIG_PATH="$2"
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
      echo "  --clients N        Number of clients to run (default: 5)"
      echo "  --delay N          Delay between client startups in seconds (default: 10)"
      echo "  --server-wait N    Time to wait for server initialization in seconds (default: 15)"
      echo "  --config PATH      Path to config file (default: config/default.yaml)"
      echo "  --port N           Server port (default: 8000)"
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

# Configuration from docker-compose.yaml
SERVER_URL="http://localhost:$SERVER_PORT"
CUDA_VISIBLE_DEVICES="-1"
CLIENT_TYPE="standard"
TF_NUM_THREADS="4"
TF_CPP_MIN_LOG_LEVEL="1"

# Directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$DIR/.venv"

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
        env_vars="export CLIENT_ID=\"$client_id\" && "
        env_vars+="export SERVER_URL=\"$SERVER_URL\" && "
        env_vars+="export CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" && "
        env_vars+="export CONFIG_PATH=\"$CONFIG_PATH\" && "
        env_vars+="export CLIENT_TYPE=\"$CLIENT_TYPE\" && "
        env_vars+="export TF_NUM_THREADS=\"$TF_NUM_THREADS\" && "
        env_vars+="export TF_CPP_MIN_LOG_LEVEL=\"$TF_CPP_MIN_LOG_LEVEL\""
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
echo -e "  Number of clients: ${YELLOW}$NUM_CLIENTS${NC}"
echo -e "  Delay between clients: ${YELLOW}$DELAY_BETWEEN seconds${NC}"
echo -e "  Config file: ${YELLOW}$CONFIG_PATH${NC}"
echo -e "  Server port: ${YELLOW}$SERVER_PORT${NC}"
echo -e "  Terminal type: ${YELLOW}$TERMINAL_TYPE${NC}"
echo ""

# Start the server
echo -e "${BLUE}Starting server on port $SERVER_PORT...${NC}"
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
echo -e "  ${BLUE}./run_federated.sh --clients 10 --delay 5${NC}"
echo -e "  ${BLUE}./run_federated.sh --help${NC} (for all options)"