CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -pedantic
TARGET = dijkstra_sequence
SRC = src/main.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

run-sample: $(TARGET)
	./$(TARGET) < tests/sample.in

pdf:
	pandoc ../docs/report.md -o ../docs/report.pdf \
		--from markdown+raw_tex \
		--pdf-engine=xelatex \
		-V papersize:a4 \
		-V geometry:margin=1in \
		-V CJKmainfont='PingFang SC'

clean:
	rm -f $(TARGET)

.PHONY: all run-sample pdf clean
