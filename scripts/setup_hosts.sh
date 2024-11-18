#!/bin/bash
set -e

echo "Configuring hosts for distributed training..."

# Create hosts file
cat > hostfile << EOF
primary slots=1
secondary slots=1
EOF

# Configure SSH access
for host in $(cat hostfile | cut -d' ' -f1); do
    # Test SSH connection
    ssh -o StrictHostKeyChecking=no $host "hostname" || {
        echo "Failed to connect to $host"
        exit 1
    }
done

# Test MPI connectivity
mpirun --hostfile hostfile -np 2 hostname

# Test network speed
if command -v iperf3 &> /dev/null; then
    echo "Testing network speed..."
    iperf3 -c secondary-device.local -t 10
fi

echo "Host configuration complete!"