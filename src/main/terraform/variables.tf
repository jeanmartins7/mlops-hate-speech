variable "aws_region" {
  description = "Região da AWS para implantar os recursos."
  type        = string
  default     = "us-east-1"
}

variable "hate_speech" {
  description = "Nome do projeto para prefixar recursos."
  type        = string
  default     = "mlops-hate-speech"
}