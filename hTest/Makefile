action:
	@echo action $(filter-out $@,$(MAKECMDGOALS))

%: %.c
	gcc $< -o ./compiled/$@ -L/usr/lib -lgsl -lgslcblas -lm && ./compiled/$@