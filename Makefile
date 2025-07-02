PKGS = numpy pygame

GREY   = \033[30m
GREEN  = \033[32m
YELLOW = \033[33m
RESET  = \033[0m\n

.SILENT:

all: check
	printf "$(GREEN) Packages Ready! $(RESET)"
	printf "$(GREY)  Usage:$(YELLOW) make {t, v, s} $(RESET)"

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg > /dev/null 2>&1; \
		fi; \
	done

t:
	python3 src/game.py -load 'models/good.pth' -sessions 100

d:
	python3 src/game.py -save 'models/100sess.pth' -sessions 100
	# python3 src/game.py -load 'models/dist.pth' -save 'models/dist.pth' -sessions 200 -fps 2000

v:
	python3 src/game.py -load 'models/dist.pth' -vn -fps 7

s:
	python3 src/game.py -load 'models/dist.pth' -vns -fps 7

clean:
	find . \( -name "__pycache__" -o -name ".DS_Store" \) -print0 | xargs -0 rm -rf
	rm -rf data/train.csv data/val.csv

fclean: clean
	rm -rf net.pkl
	find . -name .DS_Store -delete

gpush: fclean
	git add .
	git commit -m "models"
	git push

re: fclean all
