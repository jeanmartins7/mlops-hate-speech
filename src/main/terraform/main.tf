
data "aws_vpc" "default_vpc"{
  default = true
}

data "aws_ami" "amazon_linux_2" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
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
  ami                    = data.aws_ami.amazon_linux_2.id
  instance_type          = "t2.micro"
  key_name               = aws_key_pair.mlops_key_pair.key_name
  security_groups        = [aws_security_group.ec2_security_group.name]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

user_data = <<-EOF
              #!/bin/bash
              set -e

              echo "Atualizando pacotes do sistema..."
              sudo yum update -y
              sudo yum install -y python3-pip git -y

              echo "Instalando bibliotecas Python via pip3..."
              sudo pip3 install --upgrade pip

              # Usamos sudo e especificamos versões para maior compatibilidade com Amazon Linux 2 (Python 3.7/3.8)
              sudo python3 -m pip install \
                  pandas==1.3.5 \
                  numpy==1.21.6 \
                  scikit-learn==1.0.2 \
                  torch==1.13.1 \
                  transformers==4.26.1 \
                  datasets==2.13.2 \
                  nltk==3.8.1 \
                  --no-cache-dir # Evita problemas com cache

              echo "Baixando modelos NLTK..."
              sudo python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

              echo "Instalação e configuração concluídas."
              EOF

  tags = {
    Name        = "${var.hate_speech}-compute-instance"
    Project     = var.hate_speech
    Environment = "dev"
  }
}