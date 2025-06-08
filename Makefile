PKGS = numpy

GREY   = \033[30m
GREEN  = \033[32m
YELLOW = \033[33m
RESET  = \033[0m\n

.SILENT:

all: check
	printf "$(GREEN) Packages Ready! $(RESET)"
	printf "$(GREY)  Usage:$(YELLOW) make {gen, t, p} $(RESET)"

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg > /dev/null 2>&1; \
		fi; \
	done

a:
	python3 src/game.py -load 'models/first.pth' -size 10 20 -n

t:
	# python3 src/game.py -save 'models/first.pth' -sessions 20 -fps 200
	python3 src/game.py -load 'models/first.pth' -save 'models/first.pth' -sessions 2000 -fps 2000

d:
	# python3 src/game.py -save 'models/dist.pth' -sessions 20 -fps 200
	python3 src/game.py -load 'models/dist.pth' -save 'models/dist.pth' -sessions 2000 -fps 2000

v:
	# python3 src/game.py -load 'models/first.pth' -vn -fps 7
	python3 src/game.py -load 'models/dist.pth' -vn -fps 7

s:
	python3 src/game.py -load 'models/first.pth' -vns -fps 7

clean:
	find . \( -name "__pycache__" -o -name ".DS_Store" \) -print0 | xargs -0 rm -rf
	rm -rf data/train.csv data/val.csv

fclean: clean
	rm -rf net.pkl
	find . -name .DS_Store -delete

gpush: fclean
	git add .
	git commit -m "Cleaning"
	git push

re: fclean all
