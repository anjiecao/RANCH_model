# Systematic linking-hypothesis comparison figures: behavior overlaid with each
# decision variable's scaled predictions.
#   figV1: Exp1 infants, 2 rows (best CV fit | closest to human amplitude) x 4 metrics
#   figV2: Exp1 adults, 1 x 4 metrics (best R^2 settings)
#   figV3: Exp2 infants, one panel, all metrics overlaid (carried from Exp-1)
#   figV4: Exp2 adults, 1 x 4 metrics (carried), colored by violation
suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)
  library(ggthemes); library(cowplot)})
PAPER <- "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
GF <- "/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast"
FIGS <- "/Users/mcfrank/Projects/ranch/writeup/figs"
MET_LEVELS <- c("implemented EIG", "KL", "true EIG", "surprisal")
COL_TT <- c("familiar"="#268bd2","novel"="#cb4b16")
theme_p <- function(sz=13) theme_few(sz) + theme(strip.text=element_text(size=11),
  plot.title=element_text(size=12), legend.position="bottom")

## ---------------- figV1: Exp1 infants ----------------
beh <- suppressWarnings(read_csv(file.path(PAPER,"data/results_plots/exp1_data_plot.csv"),
                                 show_col_types=FALSE, guess_max=50000)) %>% mutate(tt=tolower(tt))
ibm <- beh %>% filter(group=="Infants (7 - 9 months)") %>% group_by(tt,x) %>%
  summarise(m=mean(y), s=sd(y), nn=n(), .groups="drop") %>%
  mutate(tt=ifelse(x==0,"familiar",tt), y=m, lo=m-1.96*s/sqrt(nn), hi=m+1.96*s/sqrt(nn))
mv <- read_csv(file.path(GF,"exp1_infant_variants.csv"), show_col_types=FALSE) %>%
  mutate(tt=tolower(test_type), x=fam_duration,
         metric=factor(metric, levels=MET_LEVELS),
         sel=factor(sel, levels=c("best CV fit","closest to human amplitude")))
# per (metric, sel): scale model to behavior over matched condition means
mv <- mv %>% group_by(metric, sel) %>% group_modify(function(d, key){
  j <- ibm %>% inner_join(d, by=c("tt","x"))
  f <- lm(y ~ mean_sample, data=j)
  d$y <- predict(f, newdata=d)
  d
}) %>% ungroup()
pV1 <- ggplot() +
  geom_pointrange(data=ibm, aes(x, y, ymin=lo, ymax=hi, group=tt),
                  color="grey35", size=.25, position=position_dodge(.4)) +
  geom_line(data=mv %>% filter(!(tt=="novel" & x==0)), aes(x, y, color=tt), linewidth=1) +
  geom_point(data=mv %>% filter(!(tt=="novel" & x==0)), aes(x, y, color=tt), size=1.6) +
  facet_grid(sel ~ metric) +
  scale_color_manual(values=COL_TT, name="Model test trial") +
  scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_p() + labs(x="Prior exposures", y="Looking time (sec)",
                   title="Experiment 1, infants: behaviour (grey, 95% CI) vs. each decision variable (scaled)")
ggsave(file.path(FIGS,"figV1_exp1_infant_variants.png"), pV1, width=13, height=6.4, dpi=150, bg="white")

## ---------------- figV2: Exp1 adults ----------------
ab <- beh %>% filter(group=="Adults") %>% group_by(tt,x) %>%
  summarise(m=mean(y), s=sd(y), nn=n(), .groups="drop") %>%
  mutate(y=m, lo=m-1.96*s/sqrt(nn), hi=m+1.96*s/sqrt(nn)) %>% rename(trial_number=x)
av <- read_csv(file.path(GF,"exp1_adult_variants.csv"), show_col_types=FALSE) %>%
  mutate(metric=factor(metric, levels=MET_LEVELS), tt=trial_type)
av <- av %>% group_by(metric) %>% group_modify(function(d, key){
  j <- ab %>% inner_join(d, by=c("tt"="tt","trial_number"="trial_number"))
  f <- lm(y ~ mean_sample, data=j)
  d$y <- predict(f, newdata=d)
  d
}) %>% ungroup()
pV2 <- ggplot() +
  geom_pointrange(data=ab, aes(trial_number, y, ymin=lo, ymax=hi, group=tt),
                  color="grey35", size=.25) +
  geom_line(data=av, aes(trial_number, y, color=tt), linewidth=1) +
  geom_point(data=av, aes(trial_number, y, color=tt), size=1.6) +
  facet_grid(. ~ metric) +
  scale_color_manual(values=COL_TT, name="Model test trial") +
  scale_x_continuous(breaks=c(1,3,6,9,11)) +
  theme_p() + labs(x="Trial number", y="Looking time (sec)",
                   title="Experiment 1, adults: behaviour (grey, 95% CI) vs. each decision variable (scaled, best-fit setting)")
ggsave(file.path(FIGS,"figV2_exp1_adult_variants.png"), pV2, width=13, height=3.9, dpi=150, bg="white")

## ---------------- figV3: Exp2 infants ----------------
e2i <- read_csv(file.path(PAPER,"data/results_plots/exp2_infant_plot.csv"), show_col_types=FALSE)
b2 <- e2i %>% filter(value_type=="Infant Behavior") %>%
  transmute(trial_type, y=LT, lo=lb_lt, hi=ub_lt)
m2 <- read_csv(file.path(GF,"exp2_infant_variants.csv"), show_col_types=FALSE) %>%
  mutate(metric=factor(metric, levels=MET_LEVELS))
m2 <- m2 %>% group_by(metric) %>% group_modify(function(d, key){
  j <- b2 %>% inner_join(d, by="trial_type")
  f <- lm(y ~ mean_sample, data=j)
  d$y <- predict(f, newdata=d)
  d
}) %>% ungroup()
ord <- c("familiar","pose","number","identity","animacy")
COL_M <- c("implemented EIG"="#2aa198","KL"="#b58900","true EIG"="#6c71c4","surprisal"="#d33682")
pV3 <- ggplot() +
  geom_pointrange(data=b2 %>% mutate(trial_type=factor(trial_type, levels=ord)),
                  aes(trial_type, y, ymin=lo, ymax=hi), color="black", size=.4) +
  geom_line(data=m2 %>% mutate(trial_type=factor(trial_type, levels=ord)),
            aes(trial_type, y, color=metric, group=metric), linewidth=1) +
  geom_point(data=m2 %>% mutate(trial_type=factor(trial_type, levels=ord)),
             aes(trial_type, y, color=metric), size=2) +
  scale_color_manual(values=COL_M, name="") +
  theme_p(14) + labs(x="Violation type (increasing dissimilarity)", y="Looking time (sec)",
                     title="Experiment 2, infants: behaviour (black, 95% CI) vs. decision variables carried from Exp-1")
ggsave(file.path(FIGS,"figV3_exp2_infant_variants.png"), pV3, width=8.2, height=5.2, dpi=150, bg="white")

## ---------------- figV4: Exp2 adults ----------------
e2a <- read_csv(file.path(PAPER,"data/results_plots/exp2_adult_plot.csv"), show_col_types=FALSE)
b2a <- e2a %>% filter(value_type=="Adult Behavior") %>% transmute(trial_type, trial_number, y=LT)
m2a <- read_csv(file.path(GF,"exp2_adult_variants.csv"), show_col_types=FALSE) %>%
  mutate(metric=factor(metric, levels=MET_LEVELS))
m2a <- m2a %>% group_by(metric) %>% group_modify(function(d, key){
  j <- b2a %>% inner_join(d, by=c("trial_type","trial_number"))
  f <- lm(y ~ mean_sample, data=j)
  d$y <- predict(f, newdata=d)
  d
}) %>% ungroup()
COL_V <- c("fam"="grey40","pose"="#268bd2","number"="#2aa198","identity"="#6c71c4","animacy"="#cb4b16")
pV4 <- ggplot() +
  geom_point(data=b2a, aes(trial_number, y, color=trial_type), shape=1, size=2.2, stroke=.9) +
  geom_line(data=m2a, aes(trial_number, y, color=trial_type), linewidth=1) +
  geom_point(data=m2a, aes(trial_number, y, color=trial_type), size=1.6) +
  facet_grid(. ~ metric) +
  scale_color_manual(values=COL_V, name="", breaks=c("fam","pose","number","identity","animacy")) +
  scale_x_continuous(breaks=c(1,2,4,6)) +
  theme_p() + labs(x="Position in block", y="Looking time (ms)",
                   title="Experiment 2, adults: behaviour (open circles) vs. decision variables carried from Exp-1 (lines)")
ggsave(file.path(FIGS,"figV4_exp2_adult_variants.png"), pV4, width=13, height=4.1, dpi=150, bg="white")

cat("wrote figV1..figV4\n")
