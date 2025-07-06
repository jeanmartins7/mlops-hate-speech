
data "aws_vpc" "default_vpc"{
  default = true
}

data "aws_ami" "amazon_linux_latest" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-kernel-6.1-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.hate_speech}-data-bucket-${lower(var.aws_region)}"

  tags = {
    Project     = var.hate_speech
    Environment = "dev"
  }
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  bucket = aws_s3_bucket.data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_iam_role" "ec2_s3_access_role" {
  name = "${var.hate_speech}-ec2-s3-access-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
      },
    ],
  })

  tags = {
    Project     = var.hate_speech
    Environment = "dev"
  }
}

resource "aws_iam_role_policy" "ec2_s3_policy" {
  name = "${var.hate_speech}-ec2-s3-policy"
  role = aws_iam_role.ec2_s3_access_role.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ],
        Effect   = "Allow",
        Resource = [
          aws_s3_bucket.data_bucket.arn,
          "${aws_s3_bucket.data_bucket.arn}/*",
        ],
      },
      {
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Effect   = "Allow",
        Resource = "arn:aws:logs:*:*:*"
      },
    ],
  })
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.hate_speech}-ec2-profile"
  role = aws_iam_role.ec2_s3_access_role.name
}

resource "aws_key_pair" "mlops_key_pair" {
  key_name   = "${var.hate_speech}-ec2-key"
  public_key = file("../resource/env/ec2/pem/id_rsa_terraform-ec2-key.pub")
}


resource "aws_security_group" "ec2_security_group" {
  name        = "${var.hate_speech}-ec2-sg"
  description = "Permitir trafego SSH para a instancia EC2"
  vpc_id      = data.aws_vpc.default_vpc.id

  ingress {
    description = "SSH from VPC"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.hate_speech
    Environment = "dev"
  }
}

resource "aws_instance" "mlops_compute_instance" {
  ami                    = data.aws_ami.amazon_linux_latest.id
  instance_type          = "t2.micro"
  key_name               = aws_key_pair.mlops_key_pair.key_name
  security_groups        = [aws_security_group.ec2_security_group.name]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_size = 30
    volume_type = "gp2"
    delete_on_termination = true
  }

      user_data = <<-EOF
              #!/bin/bash
              set -e # Sai imediatamente se qualquer comando falhar

              exec > >(tee /var/log/custom-user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

              echo "Iniciando configuração de dependências Python..."

              echo "Atualizando pacotes do sistema..."
              sudo timeout 300 bash -c "while fuser /var/run/yum.pid >/dev/null 2>&1; do echo 'Waiting for another yum process to finish...'; sleep 10; done; yum update -y"

              echo "Instalando git e python3-pip..."
              sudo yum install -y python3-pip git -y

              echo "Instalando bibliotecas Python via pip3..."
              sudo pip3 install --upgrade pip

              sudo python3 -m pip install \
                  urllib3==1.26.18 \
                  requests==2.25.1 \
                  pandas==1.3.5 \
                  numpy==1.21.6 \
                  scikit-learn==1.0.2 \
                  torch==1.13.1 \
                  transformers==4.26.1 \
                  datasets==2.13.2 \
                  nltk==3.8.1 \
                  huggingface-hub==0.11.1 \
                  --no-cache-dir

              echo "Verificando se as principais bibliotecas foram instaladas..."
              python3 -c "import pandas; import numpy; import sklearn; import transformers; import torch; import datasets; import nltk; print('PYTHON_LIBS_INSTALLED_SUCCESSFULLY')"

              if [ $? -eq 0 ]; then
                  echo "Bibliotecas Python verificadas com sucesso. Prosseguindo com downloads NLTK."
                  echo "Baixando modelos NLTK..."
                  sudo python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
                  echo "Downloads NLTK concluídos."
              else
                  echo "ERRO: Falha na verificação das bibliotecas Python."
                  exit 1
              fi

              echo "Configuração de dependências concluída com sucesso!"
              EOF

  tags = {
    Name        = "${var.hate_speech}-compute-instance"
    Project     = var.hate_speech
    Environment = "dev"
  }
}