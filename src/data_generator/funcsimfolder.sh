#!/bin/bash

# 定义所有需要处理的包名
PACKAGES="
argon2
bind
bird
cairo
capnproto
chrony
cmake
containerd
cryptsetup
curl
dbus
dhcp
dovecot
dpdk
elastic
expat
flatbuffers
fontconfig
gdb
gettext
gnutls
gpgme
harfbuzz
icu
inkscape
iptables
istio
keepalived
libgcrypt
libiconv
libressl
libsodium
libxslt
lz4
make
memcached
mercurial
minio
nftables
nmap
nss
ntp
openldap
openssh
openssl
openvpn
openvswitch
pango
pixman
postfix
postgresql
prometheus
python
runc
scrypt
socat
strace
strongswan
stunnel
tcpdump
terraform
wayland
wget
wireshark
xxhash
zlib
zstd
"

# 创建目标父目录
TARGET_DIR="/data/kun/funcsim_dataset"
mkdir -p "$TARGET_DIR"

# 开始循环处理
for pkg in $PACKAGES; do
    # 针对每个包，循环处理 O0 到 O3
    for i in {0..3}; do
        src_path="/data/kun/elf_O$i/$pkg"
        dest_path="$TARGET_DIR/${pkg}_O$i"

        # 检查源目录是否存在，防止报错
        if [ -d "$src_path" ]; then
            echo "Processing $pkg (O$i)..."
            # 递归复制并将目录重命名为 pkg_Oi
            cp -r "$src_path" "$dest_path"
        else
            echo "⚠️  Warning: Source not found for $src_path, skipping..."
        fi
    done
done

echo "✅ All Done!"