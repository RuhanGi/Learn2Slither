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
	python3 src/main.py -size 5 5

t:
	# python3 src/main.py -save 'models/first.pth' -max 20
	python3 src/main.py -load 'models/first.pth' -save 'models/first.pth' -max 20

v:
	python3 src/game.py

clean:
	find . \( -name "__pycache__" -o -name ".DS_Store" \) -print0 | xargs -0 rm -rf
	rm -rf data/train.csv data/val.csv

fclean: clean
	rm -rf net.pkl
	find . -name .DS_Store -delete

gpush: fclean
	git add .
	git commit -m "VISUALS"
	git push

re: fclean all
