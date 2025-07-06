output "data_bucket_hate_speach" {
  description = "Nome do bucket S3 para os dados."
  value       = aws_s3_bucket.data_bucket.bucket
}

output "data_bucket_arn" {
  description = "ARN do bucket S3 para os dados."
  value       = aws_s3_bucket.data_bucket.arn
}

output "ec2_public_ip" {
  description = "Endereço IP público da instância EC2."
  value       = aws_instance.mlops_compute_instance.public_ip
}

output "ec2_instance_id" {
  description = "ID da instância EC2."
  value       = aws_instance.mlops_compute_instance.id
}
