variable "aws_region" {
  description = "AWS region to deploy resources in."
  type        = string
  default     = "us-west-1"
}

variable "ec2_instance_type" {
  description = "EC2 instance type for MLflow server."
  type        = string
  default     = "t3.small"
}

variable "ec2_ssh_key_name" {
  description = "SSH key pair name for EC2 access. Must already exist in AWS."
  type        = string
  default     = ""
}

variable "allowed_cidr" {
  description = "CIDR block allowed to access SSH and MLflow UI. Default is your current IP."
  type        = string
  default     = "0.0.0.0/0"
}

variable "s3_bucket_name" {
  description = "S3 bucket name for MLflow artifacts."
  type        = string
  default     = "model-data-bucket-4821"
}

variable "s3_artifact_prefix" {
  description = "S3 prefix/folder for MLflow artifacts."
  type        = string
  default     = "mlflow-artifacts/"
}

variable "tags" {
  description = "Tags to apply to all resources."
  type        = map(string)
  default     = {
    Project = "mlflow-server"
  }
} 