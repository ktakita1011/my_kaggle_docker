FROM gcr.io/kaggle-gpu-images/python

ENV lang="ja_jp.utf-8" language="ja_jp:ja" lc_all="ja_jp.utf-8"

RUN jupyter notebook --generate-config && \
	sed -i 's/#c.NotebookApp.quit_button = True/c.NotebookApp.quit_button = False/' /root/.jupyter/jupyter_notebook_config.py

ADD run.sh /opt/run.sh
RUN chmod 700 /opt/run.sh

WORKDIR /workdir
ADD requirements.txt /workdir/requirements.txt
RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install -r requirements.txt

CMD /opt/run.sh