sudo apt install python3 python3-pip -y
sudo apt install python3-venv -y
source venv/bin/activate
pip install flask ddgs beautifulsoup4 requests sentence-transformers torch gunicorn aiohttp --no-cache-dir
