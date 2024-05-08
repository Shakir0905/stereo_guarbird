FROM stereolabs/zed:4.0-runtime-cuda12.1-ubuntu22.04 AS dependencies

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip
RUN pip3 install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

FROM dependencies AS runtime

COPY . /app

CMD ["python3", "main.py"]
