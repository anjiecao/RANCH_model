# Experiment 2 comparison figures: behaviour vs original grid RANCH vs corrected.
# Fig 6 (infants): looking time by violation type (increasing dissimilarity).
# Fig 7 (adults):  familiar habituation + violation tests at positions 2/4/6.

suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2); library(ggthemes); library(cowplot)})
PAPER <- "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
GF <- "/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast"
FIGS <- "/Users/mcfrank/Projects/ranch/writeup/figs"
VORD <- c("familiar","pose","number","identity","animacy")
scale_pred <- function(model, lt){ f<-lm(lt~model); coef(f)[1]+coef(f)[2]*model }

## ---------- FIG 6: INFANT Exp 2 (behaviour vs grid vs corrected) ----------
e2i <- read_csv(file.path(PAPER,"data/results_plots/exp2_infant_plot.csv"), show_col_types=FALSE)
beh <- e2i %>% filter(value_type=="Infant Behavior") %>% select(trial_type, LT, lb_lt, ub_lt)
grid <- e2i %>% filter(value_type=="RANCH") %>% select(trial_type, LT)              # paper's RANCH (grid)
mod <- read_csv(file.path(GF,"exp2_infant_eigcode.csv"), show_col_types=FALSE)     # corrected raw samples
mm <- beh %>% select(trial_type, LT) %>% inner_join(mod, by="trial_type")
mod$LT <- scale_pred(mod$mean_sample, mm$LT[match(mod$trial_type, mm$trial_type)])
d6 <- bind_rows(
  beh %>% transmute(trial_type, value="Infant behaviour", LT, lo=lb_lt, hi=ub_lt),
  grid %>% transmute(trial_type, value="RANCH: grid (original)", LT, lo=NA, hi=NA),
  mod %>% transmute(trial_type, value="RANCH: exact (paper EIG)", LT, lo=NA, hi=NA)
) %>% mutate(trial_type=factor(trial_type, levels=VORD))
p6 <- ggplot(d6, aes(trial_type, LT, color=value, group=value)) +
  geom_line(linewidth=1) + geom_pointrange(aes(ymin=lo,ymax=hi), size=.6) +
  scale_color_manual(values=c("Infant behaviour"="black","RANCH: grid (original)"="#b58900","RANCH: exact (paper EIG)"="#2aa198"), name="") +
  theme_few(14) + labs(x="Violation type (increasing dissimilarity →)", y="Looking time (sec)",
                       title="Experiment 2 (infants): graded dishabituation by violation type") +
  theme(legend.position=c(0.25,0.86))
ggsave(file.path(FIGS,"fig6_infant_exp2_eigcode.png"), p6, width=8, height=5, dpi=150, bg="white")

## ---------- FIG 7: ADULT Exp 2 (behaviour | grid | corrected) ----------
e2a <- read_csv(file.path(PAPER,"data/results_plots/exp2_adult_plot.csv"), show_col_types=FALSE)
modA <- read_csv(file.path(GF,"exp2_adult_eigcode.csv"), show_col_types=FALSE)
behA <- e2a %>% filter(value_type=="Adult Behavior")
gridA <- e2a %>% filter(value_type=="RANCH")
# scale corrected to adult RT via matched (trial_type, trial_number) vs behaviour
mA <- behA %>% select(trial_type, trial_number, LT) %>% inner_join(modA, by=c("trial_type","trial_number"))
modA$LT <- scale_pred(modA$mean_sample, mA$LT[match(paste(modA$trial_type,modA$trial_number),
                                                    paste(mA$trial_type,mA$trial_number))])
VCOL <- c("fam"="grey40","pose"="#268bd2","number"="#2aa198","identity"="#6c71c4","animacy"="#cb4b16")
panel <- function(df, ttl){
  fam <- df %>% filter(trial_type=="fam"); dev <- df %>% filter(trial_type!="fam")
  ggplot() +
    geom_line(data=fam, aes(trial_number, LT), color="grey40", linewidth=1) +
    geom_point(data=fam, aes(trial_number, LT), color="grey40", size=2) +
    geom_point(data=dev, aes(trial_number, LT, color=trial_type), size=3) +
    geom_line(data=dev, aes(trial_number, LT, color=trial_type), linewidth=.8) +
    scale_color_manual(values=VCOL, name="Violation") + scale_x_continuous(breaks=c(2,4,6)) +
    theme_few(13) + labs(x="Position in block", y="Looking time (ms)", title=ttl)
}
p7 <- plot_grid(
  panel(behA, "Adult behaviour")+theme(legend.position="none"),
  panel(gridA, "RANCH: grid (original)")+theme(legend.position="none"),
  panel(modA, "RANCH: exact (paper EIG)")+theme(legend.position=c(0.7,0.8), legend.key.size=unit(.35,"cm")),
  nrow=1)
ggsave(file.path(FIGS,"fig7_adult_exp2_eigcode.png"), p7, width=15, height=4.4, dpi=150, bg="white")
cat("wrote fig6_infant_exp2_eigcode.png and fig7_adult_exp2_eigcode.png\n")
