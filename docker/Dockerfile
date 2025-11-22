FROM python:3.10

WORKDIR /workspace

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt

# ソースコードをコピー（必要に応じて）
COPY . .

CMD ["/bin/bash"]