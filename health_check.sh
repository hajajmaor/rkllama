#!/bin/bash

# RKLLAMA Health Check Script
# Runs every hour via cron to check model health and restart if needed

# Configuration
OLLAMA_HOST="localhost"
OLLAMA_PORT="8080"
MODEL_NAME="qwen:4b"
TIMEOUT=60
CONTAINER_NAME="rkllama"
N8N_WEBHOOK_URL="https://n8n.hajaj-projects.com/webhook/rkllama-health-check"

# Function to send webhook notification
send_webhook() {
    local status="$1"
    local action="$2"
    local message="$3"
    
    local payload=$(cat <<EOF
{
    "status": "$status",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "action": "$action",
    "message": "$message",
    "model": "$MODEL_NAME",
    "container": "$CONTAINER_NAME"
}
EOF
)
    
    # Send webhook notification with JSON payload
    echo "Sending webhook notification..."
    curl --location --request POST "$N8N_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        --max-time 10 \
        -w "Webhook HTTP Status: %{http_code}\n"
}

# Function to perform health check
health_check() {
    local request_payload=$(cat <<EOF
{
    "model": "$MODEL_NAME",
    "prompt": "Please respond with the word OK",
    "stream": false
}
EOF
)
    
    # Perform health check with timeout
    local response
    response=$(curl -s --max-time $TIMEOUT -X POST "http://$OLLAMA_HOST:$OLLAMA_PORT/api/generate" \
        -H "Content-Type: application/json" \
        -d "$request_payload" 2>/dev/null)
    
    local exit_code=$?
    
    # Check if request was successful (we don't care about the actual response content)
    if [ $exit_code -eq 0 ] && [ -n "$response" ]; then
        return 0  # Success
    else
        return 1  # Failed
    fi
}

# Function to restart RKLLAMA
restart_rkllama() {
    echo "Health check failed. Restarting RKLLAMA..."
    
    # Step 1: Unload the model
    echo "Unloading model..."
    /home/kono/RKLLAMA/client.py unload 2>/dev/null || true
    
    # Step 2: Stop the container
    echo "Stopping container..."
    docker-compose -f /home/kono/RKLLAMA/docker-compose.yml stop rkllama
    
    # Step 3: Wait 10 seconds
    echo "Waiting 10 seconds..."
    sleep 10
    
    # Step 4: Start the container
    echo "Starting container..."
    docker-compose -f /home/kono/RKLLAMA/docker-compose.yml start rkllama
    
    echo "Restart completed."
}

# Main execution
main() {
    echo "Starting RKLLAMA health check at $(date)"
    
    # Perform health check
    if health_check; then
        echo "Health check passed"
        send_webhook "success" "health_check" "Health check passed successfully"
    else
        echo "Health check failed"
        send_webhook "failed" "health_check_failed" "Health check failed, initiating restart"
        
        # Restart RKLLAMA
        restart_rkllama
        
        # Send restart notification
        send_webhook "success" "restart_performed" "RKLLAMA container restarted due to failed health check"
    fi
    
    echo "Health check completed at $(date)"
}

# Run main function
main "$@"
