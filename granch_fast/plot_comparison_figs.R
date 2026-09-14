# Side-by-side comparison figures for the report: behaviour | original grid RANCH
# | corrected RANCH. Grid panels use the paper's OWN plotting data (already scaled
# to looking time); corrected panels use our predictions, scaled the same way.
# All panels within a figure share the y-axis so they are directly comparable.

suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)
  library(ggthemes); library(cowplot); library(Hmisc)})
PAPER <- "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
GF <- "/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast"
FIGS <- "/Users/mcfrank/Projects/ranch/writeup/figs"
COL <- c("familiar"="#268bd2","novel"="#cb4b16","baseline"="grey")
scale_fit <- function(model, lt) { f<-lm(lt~model); function(m) coef(f)[1]+coef(f)[2]*m }
theme_p <- function() theme_few(14) + theme(plot.title=element_text(size=13))

beh <- suppressWarnings(read_csv(file.path(PAPER,"data/results_plots/exp1_data_plot.csv"),
                                 show_col_types=FALSE, guess_max=50000)) %>% mutate(tt=tolower(tt))

## ============ FIG 4: INFANT EXP 1 (4 panels) ============
inf_beh_raw <- beh %>% filter(group=="Infants (7 - 9 months)")
inf_bmean <- inf_beh_raw %>% group_by(tt,x) %>% summarise(y=mean(y), .groups="drop")
y0 <- inf_bmean %>% filter(x==0) %>% pull(y) %>% mean()
pbeh <- inf_beh_raw %>% mutate(tt=ifelse(x==0,"baseline",tt)) %>%
  ggplot(aes(x,y,color=tt,group=tt)) +
  stat_summary(fun.data=mean_cl_boot, geom="pointrange", position=position_dodge(.3)) +
  stat_summary(geom="line", position=position_dodge(.3)) +
  geom_hline(yintercept=y0, linetype="dashed", color="grey") +
  scale_color_manual(values=COL) + scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_p() + labs(x="Prior exposures", y="Looking time (sec)", title="Infant behaviour") +
  theme(legend.position="none")

# grid EIG panel (paper's own scaled samples)
grid_inf <- read_csv(file.path(PAPER,"data/results_plots/exp1_infant_sim_plot.csv"), show_col_types=FALSE) %>%
  filter(type=="EIG") %>% mutate(tt=tolower(test_type)) %>%
  group_by(tt, fam_duration) %>% summarise(y=mean(scaled_samples), .groups="drop") %>% rename(x=fam_duration)
pgrid <- ggplot(grid_inf %>% filter(!(tt=="novel"&x==0)), aes(x,y,color=tt,group=tt)) +
  geom_point(size=2) + geom_line(linewidth=1) + scale_color_manual(values=COL) +
  scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_p() + labs(x="Prior exposures", y="Scaled samples", title="RANCH: grid (original)") +
  theme(legend.position="none")

mk_corr <- function(csv, ttl){
  m <- read_csv(csv, show_col_types=FALSE) %>% mutate(tt=tolower(test_type), x=fam_duration)
  j <- inf_bmean %>% inner_join(m, by=c("tt","x")); f <- scale_fit(j$mean_sample, j$y)
  m$y <- f(m$mean_sample)
  ggplot(m %>% filter(!(tt=="novel"&x==0)), aes(x,y,color=tt,group=tt)) +
    geom_point(size=2)+geom_line(linewidth=1)+scale_color_manual(values=COL,name="Test") +
    scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
    theme_p()+labs(x="Prior exposures", y="Scaled samples", title=ttl)
}
pc1 <- mk_corr(file.path(GF,"exp1_infant_corrected.csv"),"RANCH: corrected (best fit)")+theme(legend.position="none")
pc2 <- mk_corr(file.path(GF,"exp1_infant_corrected_tight.csv"),"RANCH: corrected (tighter concept)")+
  theme(legend.position=c(0.75,0.85), legend.key.size=unit(.35,"cm"))
ggsave(file.path(FIGS,"fig4_infant_exp1_compare.png"),
       plot_grid(pbeh,pgrid,pc1,pc2,nrow=1), width=19, height=4.2, dpi=150, bg="white")

## ============ FIG 5: ADULT EXP 1 (3 panels) ============
adu_beh_raw <- beh %>% filter(group=="Adults")
adu_bmean <- adu_beh_raw %>% group_by(tt,x) %>% summarise(y=mean(y), .groups="drop") %>% rename(trial_number=x)
pAb <- adu_beh_raw %>% ggplot(aes(x,y,color=tt,group=tt)) +
  stat_summary(fun.data=mean_cl_boot, geom="pointrange", position=position_dodge(.3)) +
  stat_summary(geom="line", position=position_dodge(.3)) +
  scale_color_manual(values=COL) + scale_x_continuous(breaks=c(1,3,6,9,11)) +
  theme_p()+labs(x="Trial number", y="Looking time (sec)", title="Adult behaviour")+theme(legend.position="none")
grid_adu <- read_csv(file.path(PAPER,"data/results_plots/exp1_adult_sim_plot.csv"), show_col_types=FALSE) %>%
  filter(type=="EIG") %>% mutate(tt=tolower(trial_type)) %>%
  group_by(tt, trial_number) %>% summarise(y=mean(scaled_samples), .groups="drop")
pAg <- ggplot(grid_adu, aes(trial_number,y,color=tt,group=tt)) + geom_point(size=1.8)+geom_line(linewidth=1) +
  scale_color_manual(values=COL)+scale_x_continuous(breaks=c(1,3,6,9,11)) +
  theme_p()+labs(x="Trial number", y="Scaled samples", title="RANCH: grid (original)")+theme(legend.position="none")
amod <- read_csv(file.path(GF,"exp1_adult_corrected.csv"), show_col_types=FALSE) %>% rename(tt=trial_type)
aj <- adu_bmean %>% inner_join(amod, by=c("tt","trial_number")); fa <- scale_fit(aj$mean_sample, aj$y)
amod$y <- fa(amod$mean_sample)
pAc <- ggplot(amod, aes(trial_number,y,color=tt,group=tt)) + geom_point(size=1.8)+geom_line(linewidth=1) +
  scale_color_manual(values=COL,name="Trial type")+scale_x_continuous(breaks=c(1,3,6,9,11)) +
  theme_p()+labs(x="Trial number", y="Scaled samples", title="RANCH: corrected")+theme(legend.position=c(0.8,0.85))
ggsave(file.path(FIGS,"fig5_adult_exp1_compare.png"),
       plot_grid(pAb,pAg,pAc,nrow=1), width=15, height=4.4, dpi=150, bg="white")

cat("wrote fig4_infant_exp1_compare.png and fig5_adult_exp1_compare.png\n")
