# Within-subject (deconfounded) infant comparison: does each model predict how an
# individual infant's looking DEVIATES from their own average? This is the
# identified, cohort-free test. Grid predictions are the paper's own model output
# (best within-subject parameter); corrected are ours. Binned for display; stats
# are the trial-level mixed-effects fits reported in the writeup.

suppressMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2); library(ggthemes)})
GF <- "/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast"
FIGS <- "/Users/mcfrank/Projects/ranch/writeup/figs"

d <- read_csv(file.path(GF, "infant_mixed_data.csv"), show_col_types=FALSE)

# subject-demean LT and each model prediction (removes cohort/subject baselines)
dm <- d %>% group_by(subject) %>%
  mutate(LT_w = LT - mean(LT),
         corr_w = corrected_pred - mean(corrected_pred),
         grid_w = grid_pred - mean(grid_pred)) %>% ungroup()

long <- bind_rows(
  dm %>% transmute(model="Corrected (new)", pred_w = scale(corr_w)[,1], LT_w),
  dm %>% transmute(model="Grid (old)",     pred_w = scale(grid_w)[,1], LT_w)
) %>% filter(is.finite(pred_w))

# bin standardized within-subject prediction into sextiles, mean within-subject LT per bin
binned <- long %>% group_by(model) %>%
  mutate(bin = ntile(pred_w, 6)) %>%
  group_by(model, bin) %>%
  summarise(x = mean(pred_w), y = mean(LT_w), se = sd(LT_w)/sqrt(n()), n=n(), .groups="drop")

p <- ggplot(binned, aes(x, y, color=model)) +
  geom_hline(yintercept=0, color="grey80") +
  geom_smooth(data=long, aes(pred_w, LT_w, color=model), method="lm", se=TRUE, linewidth=1, alpha=.15) +
  geom_pointrange(aes(ymin=y-se, ymax=y+se), size=.6) +
  scale_color_manual(values=c("Corrected (new)"="#2aa198","Grid (old)"="#dc322f"), name="Model") +
  theme_few(15) +
  labs(x="Model prediction (within-subject, z-scored)",
       y="Infant looking time\n(within-subject, sec)",
       title="Infant Exp 1: within-subject (deconfounded) prediction",
       subtitle="Corrected tracks within-infant looking; grid does not")
ggsave(file.path(FIGS, "fig_old_vs_new_infant_withinsubject.png"), p, width=7.5, height=5, dpi=150, bg="white")

# report slopes
for (m in unique(long$model)) {
  s <- long %>% filter(model==m); f <- lm(LT_w ~ pred_w, s)
  cat(sprintf("%-16s within-subject slope = %+.3f (p=%.2g), r = %+.3f\n",
              m, coef(f)[2], summary(f)$coefficients[2,4], cor(s$pred_w, s$LT_w)))
}
cat("wrote fig_old_vs_new_infant_withinsubject.png\n")
