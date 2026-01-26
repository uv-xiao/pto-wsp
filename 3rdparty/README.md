# Third-Party Dependencies

## pto-isa

PTO Tile Library - the core ISA that pto-wsp extends.

**Repository:** https://gitcode.com/uvxiao/pto-isa.git

### Setup

pto-isa is included as a git submodule. After cloning pto-wsp:

```bash
git submodule update --init --recursive
```

Or clone with submodules:

```bash
git clone --recursive https://gitcode.com/uvxiao/pto-wsp.git
```

### Updating

To update to the latest pto-isa:

```bash
cd 3rdparty/pto-isa
git pull origin master
cd ../..
git add 3rdparty/pto-isa
git commit -m "Update pto-isa submodule"
```

### Custom Path (Optional)

To use a different pto-isa location:

```bash
cmake -B build -DPTO_ISA_PATH=/path/to/pto-isa
```
