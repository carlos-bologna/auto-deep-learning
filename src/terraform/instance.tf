variable "credentials" {
  type = "map"
     default = {
       "access_key" = "XXX"
       "secret_key" = "XXX"
       "region"     = "us-east-1"
     }
}

provider "aws" {
  access_key = "${var.credentials.access_key}"
  secret_key = "${var.credentials.secret_key}"
  region     = "${var.credentials.region}"
}

resource "aws_spot_instance_request" "gpu-prod" {

  ami = "ami-0d96d570269578cd7"
  spot_price = "6.00"
  wait_for_fulfillment = true
  instance_type = "p3.8xlarge"
  subnet_id = "subnet-xxxxx"
  vpc_security_group_ids = ["sg-xxxxxx"]
  key_name = "gpu-instance"
  user_data = "${file("init.sh")}"

   connection {
    host     = coalesce(self.public_ip, self.private_ip)
    type     = "ssh"
    user     = "ubuntu"
    password = ""
    private_key = "${file("./gpu.pem")}"
  }

  tags = {
    Name	= "gpu-prod"
    OS		= "linux"
    Backup	= "nao"
    Project     = "deeplearning"
    Environment	= "PRD"
    Owner	= "Prevent"
  }

  volume_tags = {
    Name	= "gpu-prod"
    OS		= "linux"
    Backup	= "nao"
    Project     = "deeplearning"
    Environment	= "PRD"
    Owner	= "Prevent"
  }

  provisioner "local-exec" {
    command = "echo ${aws_spot_instance_request.gpu-prod.private_ip} > private_ip.txt"
  }

}
