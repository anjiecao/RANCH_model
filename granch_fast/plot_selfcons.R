# Figures for the self-consistent-noise finding (noisy world, learner infers eps).
#   figB1: infant Exp-1 curves per decision variable, NATIVE units next to behaviour
#   figB2: mechanism -- dur-8 test-trial decision-variable trajectories, 2x2 (metric x world)
#   figB3: configuration map -- hab & dishab ratios across model configurations
suppressMessages({library(dplyr); library(readr); library(ggplot2); library(ggthemes); library(stringr)})
PAPER <- "/Users/mcfrank/Projects/ranch/pkbb_paper_writing"
GF <- "/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast"
FIGS <- "/Users/mcfrank/Projects/ranch/writeup/figs"
COL_TT <- c("familiar"="#268bd2","novel"="#cb4b16")
MET <- c("implemented EIG","KL","true EIG","concept EIG","surprisal")   # true EIG = total (incl. eps); concept EIG = eps a nuisance (2026-09-16)
theme_p <- function(sz=13) theme_few(sz) + theme(strip.text=element_text(size=11),
  plot.title=element_text(size=12), legend.position="bottom")

## ---------- figB1: NATIVE units (no linking) ----------
# The unconstrained affine linking can invert patterns (implemented EIG's nominal best fit
# here has a NEGATIVE slope), so each variable's own expected-sample curves are shown next
# to behaviour with free y-axes; picks are the best sign-consistent settings.
beh <- suppressWarnings(read_csv(file.path(PAPER,"data/results_plots/exp1_data_plot.csv"),
                                 show_col_types=FALSE, guess_max=50000)) %>% mutate(tt=tolower(tt))
ibm <- beh %>% filter(group=="Infants (7 - 9 months)") %>% group_by(tt,x) %>%
  summarise(m=mean(y), s=sd(y), nn=n(), .groups="drop") %>%
  mutate(tt=ifelse(x==0,"familiar",tt), y=m, lo=m-1.96*s/sqrt(nn), hi=m+1.96*s/sqrt(nn))
bm <- read_csv(file.path(GF,"exp1_infant_selfcons_meta.csv"), show_col_types=FALSE)
lab <- setNames(sprintf("%s  (r = %+.2f)", bm$metric, bm$r), bm$metric)
mv <- read_csv(file.path(GF,"exp1_infant_selfcons.csv"), show_col_types=FALSE) %>%
  mutate(tt=tolower(test_type), x=fam_duration, panel=lab[metric], y=mean_sample)
bdf <- ibm %>% mutate(panel="infant behaviour (sec)")
levs <- c("infant behaviour (sec)", unname(lab[MET]))
mv$panel <- factor(mv$panel, levels=levs); bdf$panel <- factor(bdf$panel, levels=levs)
pB1 <- ggplot() +
  geom_pointrange(data=bdf, aes(x, y, ymin=lo, ymax=hi, color=tt),
                  size=.25, position=position_dodge(.4)) +
  geom_line(data=bdf, aes(x, y, color=tt), linewidth=.5, alpha=.5) +
  geom_line(data=mv %>% filter(!(tt=="novel" & x==0)), aes(x, y, color=tt), linewidth=1) +
  geom_point(data=mv %>% filter(!(tt=="novel" & x==0)), aes(x, y, color=tt), size=1.6) +
  facet_wrap(~panel, nrow=1, scales="free_y") +
  scale_color_manual(values=COL_TT, name="Test trial") +
  scale_x_continuous(breaks=c(0,1,2,3,4,6,8,9)) +
  theme_p() + labs(x="Prior exposures", y="looking (s)  |  E[samples] (native)",
    title="NOISY world, learner infers eps (the paper's spec made self-consistent): infant Exp-1, native-unit curves",
    subtitle="Each decision variable at its best sign-consistent setting, curves from the R=32 re-evaluation. Only the concept EIG (information about mu, sigma^2; eps a nuisance) produces habituation + dishabituation;\nthe total EIG (which also counts information about eps) does not, nor do the realized-gain variables.")
ggsave(file.path(FIGS,"figB1_selfcons_variants.png"), pB1, width=17, height=4.4, dpi=150, bg="white")

## ---------- figB2: mechanism ----------
mech <- read_csv(file.path(GF,"selfcons_mechanism.csv"), show_col_types=FALSE) %>%
  mutate(metric=factor(metric, levels=c("implemented EIG","true EIG","concept EIG")),
         world=factor(world, levels=c("noiseless world (published generation)","noisy world (sigma_true = 0.1)")),
         tt=ifelse(test_type=="background","familiar","novel"))
pB2 <- ggplot(mech, aes(t, y, color=tt, fill=tt)) +
  geom_ribbon(aes(ymin=lo, ymax=hi), alpha=.25, color=NA) +
  geom_line(linewidth=1) + geom_point(size=1.4) +
  facet_wrap(~ world + metric, nrow=2, scales="free_y",
             labeller=labeller(.multi_line=FALSE)) +
  scale_y_log10() +
  scale_color_manual(values=COL_TT, name="Test stimulus") +
  scale_fill_manual(values=COL_TT, guide="none") +
  theme_p() + labs(x="Test-trial sample number (after 8 exposures)", y="Decision variable (log scale)",
    title="Mechanism: the same learner (eps inferred, V1 a1 b0.1) in a noiseless vs. truly noisy world",
    subtitle=paste0("Top row: exact inference collapses eps; decision variables are ~12 orders of magnitude smaller (gaps = numerically negative values dropped by the log axis).\n",
                    "Bottom row: inference is self-consistent. Implemented EIG holds familiar ~ novel (novel/familiar 1.05 -> 0.87 over 15 samples); the total EIG sits on a floor\n",
                    "(familiar .073 -> .044, ratio 1.16 -> 1.01: information about eps, which the familiar supplies too); the concept EIG's familiar falls .029 -> .0004 (70x)\n",
                    "with the novel 1.2-1.4x above it."))
ggsave(file.path(FIGS,"figB2_selfcons_mechanism.png"), pB2, width=14, height=7, dpi=150, bg="white")

## ---------- figB3: configuration map ----------
cm <- read_csv(file.path(GF,"config_map.csv"), show_col_types=FALSE) %>%
  mutate(config=str_wrap(config, 38))
ord <- cm$config[c(1, 7, 3, 2, 6, 5, 4, 8)]   # human, the two that work, then the failures (row order = config_map.csv)
cm <- cm %>% mutate(config=factor(config, levels=rev(ord)),
                    kind=factor(kind, levels=c("human","works","fails")))
long <- bind_rows(cm %>% transmute(config, kind, panel="habituation ratio (fam dur10 / dur1; lower = habituates)", v=hab),
                  cm %>% transmute(config, kind, panel="dishabituation ratio (novel/fam at dur10; >1 = dishabituates)", v=dis))
refs <- bind_rows(data.frame(panel="habituation ratio (fam dur10 / dur1; lower = habituates)", x=c(1, 0.57), lt=c("none","human")),
                  data.frame(panel="dishabituation ratio (novel/fam at dur10; >1 = dishabituates)", x=c(1, 1.72), lt=c("none","human")))
pB3 <- ggplot(long, aes(v, config, color=kind)) +
  geom_vline(data=refs, aes(xintercept=x, linetype=lt), color="grey40") +
  geom_point(size=3.2) +
  facet_wrap(~panel, scales="free_x") +
  scale_color_manual(values=c("human"="black","works"="#2aa198","fails"="#cb4b16"),
                     labels=c("human"="human data","works"="produces the phenomena","fails"="fails"), name="") +
  scale_linetype_manual(values=c("none"="dotted","human"="dashed"), guide="none") +
  theme_p(12.5) + labs(x=NULL, y=NULL,
    title="Configuration map (infant Exp-1): which combinations of world-noise and decision variable produce the phenomena",
    subtitle="Dotted line = no effect (ratio 1); dashed line = human value. Each configuration at its best (sign-consistent) setting.")
ggsave(file.path(FIGS,"figB3_config_map.png"), pB3, width=12.5, height=4.6, dpi=150, bg="white")

## ---------- figB4: ADULTS under self-consistent noise, native units ----------
am <- read_csv(file.path(GF,"adult_selfcons_curves_meta.csv"), show_col_types=FALSE)
ac <- read_csv(file.path(GF,"adult_selfcons_curves.csv"), show_col_types=FALSE)
levs4 <- c("adult behaviour (s)", am$panel[match(MET, am$metric)])
ac$panel <- factor(ac$panel, levels=levs4)
pB4 <- ggplot(ac, aes(trial_number, y, color=trial_type)) +
  geom_pointrange(data=subset(ac, !is.na(lo)), aes(ymin=lo, ymax=hi), size=.25) +
  geom_line(linewidth=.9) + geom_point(size=1.4) +
  facet_wrap(~panel, nrow=1, scales="free_y") +
  scale_color_manual(values=COL_TT, name="Trial type") +
  scale_x_continuous(breaks=c(1,3,5,7,9,11)) +
  theme_p() + labs(x="Trial number (novel = deviant after n-1 familiar trials)",
    y="dwell (s)  |  realized samples (native)",
    title="ADULTS, NOISY world, learner infers eps: self-paced Exp-1, stochastic rollouts, native units",
    subtitle="Each decision variable's winner on the 32-setting grid (21-condition linking), curves and R2 from its RE-EVALUATION on 64 fresh rollouts.\nThe concept EIG keeps the novel elevated over a declining familiar; surprisal's grid R2 of .56 was selection optimism (re-evaluated .00).")
ggsave(file.path(FIGS,"figB4_selfcons_adults.png"), pB4, width=17, height=4.4, dpi=150, bg="white")
cat("wrote figB1..figB4\n")

## ---------- figB5: channel decomposition of the forward-looking EIG ----------
# rows = quantities (I_mu, I_sigma, their sum = true EIG, realized KL, implemented EIG);
# columns = configurations isolating world noise and eps inference; same prior throughout.
# facet_grid free_y: each ROW shares a y-axis across the three worlds, so absolute
# magnitudes are comparable across columns.
ch <- read_csv(file.path(GF,"channels_decomp.csv"), show_col_types=FALSE) %>%
  mutate(channel=factor(channel, levels=c("I_mu (concept mean)","I_sigma (spread & noise)","total (true EIG)",
                                           "concept EIG (eps nuisance)","KL (realized)","implemented EIG")),
         tt=ifelse(test_type=="background","familiar","novel"))
pB5 <- ggplot(ch %>% mutate(lo=pmax(lo, y/3)), aes(t, y, color=tt, fill=tt)) +
  geom_ribbon(aes(ymin=lo, ymax=hi), alpha=.25, color=NA) +
  geom_line(linewidth=.9) + geom_point(size=1.2) +
  facet_grid(channel ~ world, scales="free_y", switch="y") +
  scale_y_log10() +
  scale_color_manual(values=COL_TT, name="Test stimulus") +
  scale_fill_manual(values=COL_TT, guide="none") +
  theme_p(12) + theme(strip.placement="outside", strip.text.y.left=element_text(angle=90)) +
  labs(x="Test-trial sample number (after 8 exposures)", y="log scale; each row shares its axis across the three worlds",
       title="Where the forward-looking EIG's novelty preference comes from, and why realized gain loses it under noise",
       subtitle=paste0("Same prior (V3 a1 b0.1) throughout. Rows 1-3: the mu-channel is stimulus-blind in every world; with eps inferred (column 3)\n",
                       "the (sigma^2, eps) channel is dominated by information about eps, which the familiar supplies as well as the novel, so the total sits on a floor.\n",
                       "Row 4: the concept EIG (eps a nuisance) equals the total when eps is fixed (columns 1-2) but habituates and dishabituates with eps inferred.\n",
                       "Rows 5-6: realized KL and the implemented functional separate novel from familiar only in the noiseless world."))
ggsave(file.path(FIGS,"figB5_channels.png"), pB5, width=12.5, height=12.5, dpi=150, bg="white")
cat("wrote figB5\n")
