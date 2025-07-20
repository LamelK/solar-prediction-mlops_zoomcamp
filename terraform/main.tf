provider "aws" {
  region = "us-west-1"
}

resource "aws_s3_bucket" "model_data_bucket" {
  bucket = "model-data-bucket-4821"
}

# Get your current public IP for security group
data "http" "myip" {
  url = "https://api.ipify.org"
}

# Security group for EC2 (SSH + MLflow UI)
resource "aws_security_group" "mlflow_ec2_sg" {
  name        = "mlflow-ec2-sg"
  description = "Allow SSH and MLflow UI"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${chomp(data.http.myip.response_body)}/32"]
  }
  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["${chomp(data.http.myip.response_body)}/32"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "mlflow-ec2-sg" }
}

# Get default VPC and subnets
data "aws_vpc" "default" {
  default = true
}
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# IAM role for EC2 to access S3
resource "aws_iam_role" "mlflow_ec2_role" {
  name = "mlflow-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "mlflow_s3_policy" {
  name = "mlflow-s3-policy"
  role = aws_iam_role.mlflow_ec2_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::model-data-bucket-4821/mlflow-artifacts/*",
        "arn:aws:s3:::model-data-bucket-4821"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "mlflow_ec2_profile" {
  name = "mlflow-ec2-profile"
  role = aws_iam_role.mlflow_ec2_role.name
}

# Generate an SSH key pair for EC2
resource "tls_private_key" "mlflow_key" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "mlflow_key" {
  key_name   = "mlflow-key"
  public_key = tls_private_key.mlflow_key.public_key_openssh
}

resource "local_file" "mlflow_private_key" {
  content         = tls_private_key.mlflow_key.private_key_pem
  filename        = "${path.module}/mlflow-key.pem"
  file_permission = "0600"
}

# Get latest Ubuntu 22.04 AMI

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_instance" "mlflow_server" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.ec2_instance_type
  subnet_id                  = data.aws_subnets.default.ids[0]
  vpc_security_group_ids      = [aws_security_group.mlflow_ec2_sg.id]
  iam_instance_profile        = aws_iam_instance_profile.mlflow_ec2_profile.name
  associate_public_ip_address = true
  key_name                   = aws_key_pair.mlflow_key.key_name
  tags                       = var.tags

  user_data = <<-EOF
    #!/bin/bash
    set -e
    exec > >(tee /var/log/user-data.log | logger -t user-data) 2>&1

    echo "Starting user data script..."

    # Update package list and install required packages
    apt-get update
    apt-get install -y python3-pip python3-venv curl wget

    # Create a virtual environment for MLflow
    python3 -m venv /opt/mlflow
    source /opt/mlflow/bin/activate

    # Upgrade pip and install MLflow + boto3
    pip install --upgrade pip
    pip install mlflow boto3

    # Create MLflow data directory
    mkdir -p /opt/mlflow/data

    # Create systemd service file for MLflow
    cat << EOT > /etc/systemd/system/mlflow.service
    [Unit]
    Description=MLflow Tracking Server
    After=network.target

    [Service]
    Type=simple
    User=ubuntu
    Group=ubuntu
    WorkingDirectory=/opt/mlflow
    Environment=PATH=/opt/mlflow/bin
    Environment=AWS_REGION=${var.aws_region}
    Environment=S3_BUCKET_NAME=${var.s3_bucket_name}
    Environment=S3_ARTIFACT_PREFIX=${var.s3_artifact_prefix}
    ExecStart=/opt/mlflow/bin/mlflow server \\
      --backend-store-uri /opt/mlflow/data/mlflow.db \\
      --default-artifact-root s3://${var.s3_bucket_name}/${var.s3_artifact_prefix} \\
      --host 0.0.0.0 --port 5000
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
    EOT

    # Set ownership to ubuntu user
    chown -R ubuntu:ubuntu /opt/mlflow

    # Enable and start the MLflow service
    systemctl daemon-reload
    systemctl enable mlflow
    systemctl start mlflow

    echo "MLflow service started successfully"
    echo "Service status:"
    systemctl status mlflow --no-pager
  EOF
}


output "mlflow_server_public_ip" {
  value = aws_instance.mlflow_server.public_ip
}

output "mlflow_ui_url" {
  value = "http://${aws_instance.mlflow_server.public_ip}:5000"
}