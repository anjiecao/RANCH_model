# Recreate the paper's Experiment-1 results figures (Fig 4 infants, Fig 5 adults)
# with the CORRECTED model, and an old-vs-new comparison figure.
# Behaviour is plotted from the paper's own plotting data; the corrected model's
# raw samples are scaled to looking time by a linear fit (as in the paper).

suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)
  library(ggthemes); library(cowplot)})

PAPER <- "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
GF    <- "/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast"
FIGS  <- "/Users/mcfrank/Projects/ranch/writeup/figs"
COL   <- c("familiar"="#268bd2", "novel"="#cb4b16", "baseline"="grey")
scale_to_lt <- function(model, lt) { f <- lm(lt ~ model); coef(f)[1] + coef(f)[2]*model }

beh <- suppressWarnings(read_csv(file.path(PAPER,"data/results_plots/exp1_data_plot.csv"),
                                 show_col_types=FALSE)) %>%
  mutate(tt = tolower(tt))

## ---------- INFANTS (Fig 4) ----------
inf_beh <- beh %>% filter(group=="Infants (7 - 9 months)") %>%
  group_by(tt, x) %>% summarise(y=mean(y), .groups="drop")
inf_mod <- read_csv(file.path(GF,"exp1_infant_corrected.csv"), show_col_types=FALSE) %>%
  mutate(tt=tolower(test_type), x=fam_duration)
inf <- inf_beh %>% inner_join(inf_mod, by=c("tt","x"))
inf_mod$scaled <- scale_to_lt(inf_mod$mean_sample,
                              inf$y[match(paste(inf_mod$tt,inf_mod$x), paste(inf$tt,inf$x))])
# fit scale on matched conditions, then apply to all model conditions
fit_i <- lm(y ~ mean_sample, data=inf); inf_mod$scaled <- predict(fit_i, inf_mod)

y0 <- inf_beh %>% filter(x==0) %>% pull(y) %>% mean()
inf_beh_p <- inf_beh %>% mutate(tt = ifelse(x==0,"baseline",tt))

p_inf_beh <- ggplot(inf_beh_p, aes(x, y, color=tt, group=tt)) +
  geom_point(size=2.5) + geom_line(linewidth=1) +
  geom_hline(yintercept=y0, linetype="dashed", color="grey") +
  scale_color_manual(values=COL) + scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_few(16) + labs(x="Prior exposures", y="Looking time (sec)", title="Infant behaviour") +
  theme(legend.position="none")
p_inf_mod <- ggplot(inf_mod %>% filter(!(tt=="novel" & x==0)), aes(x, scaled, color=tt, group=tt)) +
  geom_point(size=2.5) + geom_line(linewidth=1) +
  scale_color_manual(values=COL, name="Test trial") + scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_few(16) + labs(x="Prior exposures", y="Scaled model samples", title="Corrected RANCH (best fit)") +
  theme(legend.position="none")

# tighter-concept setting (visible habituation + dishabituation)
inf_tight <- read_csv(file.path(GF,"exp1_infant_corrected_tight.csv"), show_col_types=FALSE) %>%
  mutate(tt=tolower(test_type), x=fam_duration)
inf_t <- inf_beh %>% inner_join(inf_tight, by=c("tt","x"))
fit_it <- lm(y ~ mean_sample, data=inf_t); inf_tight$scaled <- predict(fit_it, inf_tight)
p_inf_tight <- ggplot(inf_tight %>% filter(!(tt=="novel" & x==0)), aes(x, scaled, color=tt, group=tt)) +
  geom_point(size=2.5) + geom_line(linewidth=1) +
  scale_color_manual(values=COL, name="Test trial") + scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_few(16) + labs(x="Prior exposures", y="Scaled model samples", title="Corrected RANCH (tighter concept)") +
  theme(legend.position=c(0.75,0.85), legend.key.size=unit(.4,"cm"))
ggsave(file.path(FIGS,"fig4_infant_exp1_corrected.png"),
       plot_grid(p_inf_beh, p_inf_mod, p_inf_tight, nrow=1), width=15, height=4.2, dpi=150, bg="white")

## ---------- ADULTS (Fig 5) ----------
adu_beh <- beh %>% filter(group=="Adults") %>%
  group_by(tt, x) %>% summarise(y=mean(y), .groups="drop") %>% rename(trial_number=x)
adu_mod <- read_csv(file.path(GF,"exp1_adult_corrected.csv"), show_col_types=FALSE) %>%
  rename(tt=trial_type)
adu <- adu_beh %>% inner_join(adu_mod, by=c("tt","trial_number"))
fit_a <- lm(y ~ mean_sample, data=adu); adu_mod$scaled <- predict(fit_a, adu_mod)

p_adu_beh <- ggplot(adu_beh, aes(trial_number, y, color=tt, group=tt)) +
  geom_point(size=2.5) + geom_line(linewidth=1) +
  scale_color_manual(values=COL) + scale_x_continuous(breaks=1:11) +
  theme_few(16) + labs(x="Trial number", y="Looking time (sec)", title="Adult behaviour") +
  theme(legend.position="none")
p_adu_mod <- ggplot(adu_mod, aes(trial_number, scaled, color=tt, group=tt)) +
  geom_point(size=2.5) + geom_line(linewidth=1) +
  scale_color_manual(values=COL, name="Trial type") + scale_x_continuous(breaks=1:11) +
  theme_few(16) + labs(x="Trial number", y="Scaled model samples", title="Corrected RANCH") +
  theme(legend.position=c(0.8,0.8))
ggsave(file.path(FIGS,"fig5_adult_exp1_corrected.png"),
       plot_grid(p_adu_beh, p_adu_mod, nrow=1), width=10, height=4.2, dpi=150, bg="white")

## ---------- OLD vs NEW (grid EIG vs corrected) ----------
# grid EIG predictions (already scaled) from the paper's plotting data, best param by RMSE-ish: take mean over params? use their supplied EIG rows averaged per condition
grid_inf <- read_csv(file.path(PAPER,"data/results_plots/exp1_infant_sim_plot.csv"), show_col_types=FALSE) %>%
  filter(type=="EIG") %>% mutate(tt=tolower(test_type), x=fam_duration) %>%
  group_by(tt,x) %>% summarise(scaled=mean(scaled_samples), .groups="drop") %>% mutate(model="Grid (old)")
new_inf <- inf_mod %>% transmute(tt, x, scaled, model="Corrected (new)")
cmp_inf <- bind_rows(grid_inf, new_inf) %>% filter(!(tt=="novel" & x==0))
beh_inf_pts <- inf_beh %>% mutate(model="Infant behaviour")

p_cmp <- ggplot(cmp_inf, aes(x, scaled, color=tt, group=tt)) +
  geom_point(size=2) + geom_line(linewidth=1) +
  geom_point(data=inf_beh, aes(x,y,color=tt), shape=1, size=2, inherit.aes=FALSE) +
  facet_wrap(~model) + scale_color_manual(values=COL, name="Test trial") +
  scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_few(15) + labs(x="Prior exposures", y="Looking time (sec) / scaled samples",
                       title="Infant Experiment 1: old grid model vs corrected model (open circles = behaviour)")
ggsave(file.path(FIGS,"fig_old_vs_new_infant.png"), p_cmp, width=10, height=4.5, dpi=150, bg="white")

cat("wrote figures to", FIGS, "\n")
cat(sprintf("infant scale fit: R2=%.3f | adult scale fit: R2=%.3f\n",
            summary(fit_i)$r.squared, summary(fit_a)$r.squared))
