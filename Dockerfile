FROM mxmcherry/yahoo-open-nsfw-http:latest
EXPOSE 50052

RUN pip install grpcio -i http://pypi.douban.com/simple/ saltTesting --trusted-host pypi.douban.com
RUN rm -f http.py

CMD python http.py
