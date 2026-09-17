# Report v3, figures C1-C4: the paper's Figures 4-7 with three panels each --
#   behaviour | RANCH as published (the paper's own plot data) | RANCH with the concept EIG.
# The concept-EIG panel reads paper_panels_concept.csv (python -m ranch figures): native sample counts
# already mapped to seconds by the affine linking that the reported statistics use (slope >= 0).
# One y-axis per figure, so amplitudes are comparable across the three panels. Error bars: +/- 1 SE, as in the paper.

suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2); library(ggthemes); library(cowplot)})
RANCH <- Sys.getenv("RANCH_ROOT", "/Users/mcfrank/Projects/ranch")
PAPER <- file.path(RANCH, "pkbb_paper_writing/data/results_plots")
GF <- file.path(RANCH, "RANCH_model/granch_fast")
FIGS <- file.path(RANCH, "writeup/figs")
COL <- c(familiar = "#268bd2", novel = "#cb4b16", baseline = "grey55")
VCOL <- c(familiar = "#268bd2", pose = "#859900", identity = "#cb4b16", number = "#2aa198", animacy = "#d33682")
PUB <- "RANCH as published"; CON <- "RANCH with the concept EIG"
theme_p <- function() theme_few(13) + theme(plot.title = element_text(size = 12.5), legend.position = "none")
se <- function(v) sd(v) / sqrt(length(v))
concept <- read_csv(file.path(GF, "paper_panels_concept.csv"), show_col_types = FALSE)
beh1 <- suppressWarnings(read_csv(file.path(PAPER, "exp1_data_plot.csv"), show_col_types = FALSE, guess_max = 50000)) %>% mutate(tt = tolower(tt))
lines_panel <- function(d, ttl, xlab, ylab, ylim, breaks, legend = FALSE) {
  if (!"se" %in% names(d)) d$se <- NA_real_                       # Monte-Carlo SE of a model mean, where the tables carry it
  p <- ggplot(d, aes(x, y, color = tt, group = tt)) + geom_line(linewidth = .9) + geom_point(size = 2) +
    geom_linerange(aes(ymin = y - ifelse(is.na(se), 0, se), ymax = y + ifelse(is.na(se), 0, se)), linewidth = .5) +
    scale_color_manual(values = COL, name = NULL) + scale_x_continuous(breaks = breaks) + coord_cartesian(ylim = ylim) +
    theme_p() + labs(x = xlab, y = ylab, title = ttl)
  if (legend) p <- p + theme(legend.position = c(.8, .16), legend.background = element_blank(), legend.key.size = unit(.4, "cm"))
  p
}

## ---- C1 = paper Fig. 4: infants, Exp 1
ib <- beh1 %>% filter(group == "Infants (7 - 9 months)") %>% mutate(tt = ifelse(x == 0, "baseline", tt)) %>%
  group_by(tt, x) %>% summarise(se = se(y), y = mean(y), .groups = "drop")
ylim1 <- c(min(ib$y - ib$se), max(ib$y + ib$se))
p1b <- ggplot(ib, aes(x, y, color = tt, group = tt)) + geom_line(data = filter(ib, tt != "baseline"), linewidth = .6) +
  geom_pointrange(aes(ymin = y - se, ymax = y + se), position = position_dodge(.3), size = .45) +
  scale_color_manual(values = COL) + scale_x_continuous(breaks = c(0, 1, 2, 3, 4, 6, 8, 9)) + coord_cartesian(ylim = ylim1) +
  theme_p() + labs(x = "Prior exposures", y = "Looking time (s)", title = "Infant behaviour")
ip <- read_csv(file.path(PAPER, "exp1_infant_sim_plot.csv"), show_col_types = FALSE) %>% filter(type == "EIG") %>%
  mutate(tt = tolower(test_type)) %>% group_by(tt, x = fam_duration) %>% summarise(y = mean(scaled_samples), .groups = "drop")
ic <- concept %>% filter(figure == "exp1_infants", !(trial_type == "novel" & x == 0), x %in% c(0, 1, 2, 3, 4, 6, 8, 9)) %>% transmute(tt = trial_type, x, y = scaled, se = se_scaled)
ggsave(file.path(FIGS, "figC1_infant_exp1.png"), width = 10.5, height = 3.3, dpi = 160, bg = "white",
       plot_grid(p1b, lines_panel(ip, PUB, "Prior exposures", "Scaled samples (s)", ylim1, c(0, 1, 2, 3, 4, 6, 8, 9)),
                 lines_panel(ic, CON, "Prior exposures", "Scaled samples (s)", ylim1, c(0, 1, 2, 3, 4, 6, 8, 9), legend = TRUE), nrow = 1))

## ---- C2 = paper Fig. 5: adults, Exp 1
ab <- beh1 %>% filter(group == "Adults") %>% group_by(tt, x) %>% summarise(se = se(y), y = mean(y), .groups = "drop")
ap <- read_csv(file.path(PAPER, "exp1_adult_sim_plot.csv"), show_col_types = FALSE) %>% filter(type == "EIG") %>%
  mutate(tt = tolower(trial_type)) %>% group_by(tt, x = trial_number) %>% summarise(y = mean(scaled_samples) / 1000, .groups = "drop")
ac <- concept %>% filter(figure == "exp1_adults") %>% transmute(tt = trial_type, x, y = scaled, se = se_scaled)
ylim2 <- range(c(ab$y - ab$se, ab$y + ab$se, ap$y, ac$y))
p2b <- ggplot(ab, aes(x, y, color = tt, group = tt)) + geom_line(linewidth = .6) + geom_pointrange(aes(ymin = y - se, ymax = y + se), size = .4) +
  scale_color_manual(values = COL) + scale_x_continuous(breaks = c(1, 3, 6, 9, 11)) + coord_cartesian(ylim = ylim2) +
  theme_p() + labs(x = "Trial number", y = "Looking time (s)", title = "Adult behaviour")
ggsave(file.path(FIGS, "figC2_adult_exp1.png"), width = 10.5, height = 3.3, dpi = 160, bg = "white",
       plot_grid(p2b, lines_panel(ap, PUB, "Trial number", "Scaled samples (s)", ylim2, c(1, 3, 6, 9, 11)),
                 lines_panel(ac, CON, "Trial number", "Scaled samples (s)", ylim2, c(1, 3, 6, 9, 11), legend = TRUE), nrow = 1))

## ---- C3 = paper Fig. 6: infants, Exp 2
ORD <- c("familiar", "pose", "identity", "number", "animacy")
i2 <- read_csv(file.path(PAPER, "exp2_infant_plot.csv"), show_col_types = FALSE) %>%
  transmute(panel = ifelse(value_type == "RANCH", PUB, "Infant behaviour"), tt = trial_type, y = LT, lo = lb_lt, hi = ub_lt)
i2 <- bind_rows(i2, concept %>% filter(figure == "exp2_infants") %>% transmute(panel = CON, tt = trial_type, y = scaled, lo = scaled - se_scaled, hi = scaled + se_scaled)) %>%
  mutate(tt = factor(tt, ORD), panel = factor(panel, c("Infant behaviour", PUB, CON)))
p3 <- ggplot(i2, aes(tt, y, color = tt)) + geom_pointrange(aes(ymin = ifelse(is.na(lo), y, lo), ymax = ifelse(is.na(hi), y, hi)), size = .55) +
  facet_wrap(~panel, nrow = 1) + scale_color_manual(values = VCOL) + theme_p() + theme(strip.text = element_text(size = 12.5, hjust = 0)) +
  labs(x = "Violation type", y = "Looking time / scaled samples (s)")
ggsave(file.path(FIGS, "figC3_infant_exp2.png"), p3, width = 10.5, height = 3.2, dpi = 160, bg = "white")

## ---- C4 = paper Fig. 7: adults, Exp 2
a2 <- read_csv(file.path(PAPER, "exp2_adult_plot.csv"), show_col_types = FALSE) %>%
  transmute(panel = ifelse(value_type == "RANCH", PUB, "Adult behaviour"), tt = ifelse(trial_type == "fam", "familiar", trial_type),
            x = trial_number, y = LT / 1000, lo = lb_lt / 1000, hi = ub_lt / 1000)
a2 <- bind_rows(a2, concept %>% filter(figure == "exp2_adults") %>% transmute(panel = CON, tt = trial_type, x, y = scaled, lo = scaled - se_scaled, hi = scaled + se_scaled)) %>%
  mutate(tt = factor(tt, ORD), panel = factor(panel, c("Adult behaviour", PUB, CON)))
p4 <- ggplot(a2, aes(x, y, color = tt, group = tt)) + geom_line(linewidth = .6, position = position_dodge(.25)) +
  geom_pointrange(aes(ymin = ifelse(is.na(lo), y, lo), ymax = ifelse(is.na(hi), y, hi)), size = .4, position = position_dodge(.25)) +
  facet_wrap(~panel, nrow = 1) + scale_color_manual(values = VCOL, name = "Violation type") + scale_x_continuous(breaks = 1:6) +
  theme_p() + theme(strip.text = element_text(size = 12.5, hjust = 0), legend.position = "right") + labs(x = "Trial number", y = "Looking time / scaled samples (s)")
ggsave(file.path(FIGS, "figC4_adult_exp2.png"), p4, width = 10.9, height = 3.2, dpi = 160, bg = "white")

## ---- C5: why the three quantities behave differently under real noise (selfcons_mechanism.csv, noisy world only)
NAMES <- c("implemented EIG" = "published EIG (realized gain)", "true EIG" = "total EIG: about (mu, sigma, eps)", "concept EIG" = "concept EIG: about (mu, sigma)")
mech <- read_csv(file.path(GF, "selfcons_mechanism.csv"), show_col_types = FALSE) %>% filter(grepl("^noisy", world)) %>%
  mutate(panel = factor(NAMES[metric], NAMES), tt = ifelse(test_type == "background", "familiar", "novel"))
p5 <- ggplot(mech, aes(t, y, color = tt, fill = tt, group = tt)) + geom_ribbon(aes(ymin = lo, ymax = hi), alpha = .25, color = NA) +
  geom_line(linewidth = .8) + geom_point(size = 1.3) + facet_wrap(~panel, nrow = 1, scales = "free_y") + scale_y_log10() +
  scale_color_manual(values = COL, name = NULL) + scale_fill_manual(values = COL, name = NULL) + theme_p() +
  theme(strip.text = element_text(size = 11.5, hjust = 0), legend.position = "right") +
  labs(x = "Sample number within the test trial (after 8 exposures)", y = "Decision variable (nats, log scale)")
ggsave(file.path(FIGS, "figC5_mechanism.png"), p5, width = 10.9, height = 3.1, dpi = 160, bg = "white")
cat("wrote figC1_infant_exp1.png, figC2_adult_exp1.png, figC3_infant_exp2.png, figC4_adult_exp2.png, figC5_mechanism.png\n")
