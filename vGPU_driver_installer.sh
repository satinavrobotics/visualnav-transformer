TOKEN=$1
echo "token: $TOKEN"
wget https://gitlab.wigner.hu/pinter.adam/elkh-cloud/-/raw/main/NVIDIA-Linux-x86_64-525.147.05-grid.run
wget https://gitlab.wigner.hu/pinter.adam/elkh-cloud/-/raw/main/$TOKEN.zip
apt-get update
apt-get install -y build-essential
apt-get install -y unzip
sh NVIDIA-Linux-x86_64-525.147.05-grid.run --silent
unzip -P $2 $TOKEN.zip -d /etc/nvidia/ClientConfigToken
cp /etc/nvidia/gridd.conf.template /etc/nvidia/gridd.conf
sed -i "s/^ServerAddress=$/ServerAddress=api.cls.licensing.nvidia.com/g" /etc/nvidia/gridd.conf
service nvidia-gridd restart
sleep 8
nvidia-smi -q | grep -e ": GRID" -e ": Licensed"
