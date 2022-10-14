FROM gcr.io/kaggle-gpu-images/python:v121

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

ADD run.sh /opt/run.sh
RUN chmod 700 /opt/run.sh

WORKDIR /workdir
ADD requirements.txt /workdir/requirements.txt
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -r requirements.txt

CMD /opt/run.sh