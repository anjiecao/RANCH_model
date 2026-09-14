# Mixed-effects analysis of infant looking time vs RANCH model predictions.
# Compares the corrected model (exact inference, fixed epsilon) and the grid model,
# handling the between-subject duration/cohort confound via a within-between
# (Mundlak) decomposition. The WITHIN-subject coefficient is the deconfounded,
# identified estimate (it also carries the within-subject familiar-vs-novel
# dishabituation contrast, which is clean of cohort).

suppressMessages({library(lme4); library(lmerTest); library(performance)})

d <- read.csv("/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/infant_mixed_data.csv")
d$subject <- factor(d$subject)
cat(sprintf("n trials = %d, n subjects = %d\n\n", nrow(d), nlevels(d$subject)))

# standardize predictors (z-scores) so coefficients are comparable
z <- function(x) (x - mean(x)) / sd(x)
d$corr_z <- z(d$corrected_pred)
d$grid_z <- z(d$grid_pred)

# within-between (Mundlak) split for each predictor
split_wb <- function(df, col) {
  gm <- ave(df[[col]], df$subject, FUN = mean)  # subject means (between)
  list(between = gm, within = df[[col]] - gm)
}
cs <- split_wb(d, "corr_z"); d$corr_b <- cs$between; d$corr_w <- cs$within
gs <- split_wb(d, "grid_z"); d$grid_b <- gs$between; d$grid_w <- gs$within

marg <- function(m) tryCatch(as.numeric(r2_nakagawa(m)$R2_marginal), error=function(e) NA)
cond <- function(m) tryCatch(as.numeric(r2_nakagawa(m)$R2_conditional), error=function(e) NA)

cat("=========================================================\n")
cat(" Standard model:  LT ~ pred + (1|subject)   [marginal R2 = variance explained by prediction]\n")
cat("=========================================================\n")
m0 <- lmer(LT ~ 1 + (1|subject), data=d, REML=FALSE)
for (nm in c("corrected","grid")) {
  p <- if (nm=="corrected") "corr_z" else "grid_z"
  m <- lmer(reformulate(c(p, "(1|subject)"), "LT"), data=d, REML=FALSE)
  co <- summary(m)$coefficients[2,]
  lr <- anova(m0, m)
  cat(sprintf("\n%-9s: beta=%+.3f (SE %.3f), t=%.2f, p=%.2g | marginal R2=%.3f, conditional R2=%.3f | LRT vs null p=%.2g, dAIC=%.1f\n",
              nm, co[1], co[2], co[4], co[5], marg(m), cond(m), lr$`Pr(>Chisq)`[2], AIC(m0)-AIC(m)))
}

cat("\n=========================================================\n")
cat(" Within-between (Mundlak):  LT ~ pred_within + pred_between + (1|subject)\n")
cat("   pred_within  = deconfounded, identified (clean of cohort)\n")
cat("   pred_between = between-subject (confounded with sub-experiment/cohort)\n")
cat("=========================================================\n")
for (nm in c("corrected","grid")) {
  w <- if (nm=="corrected") "corr_w" else "grid_w"
  b <- if (nm=="corrected") "corr_b" else "grid_b"
  m <- lmer(reformulate(c(w, b, "(1|subject)"), "LT"), data=d, REML=FALSE)
  co <- summary(m)$coefficients
  cat(sprintf("\n%s:\n", nm))
  cat(sprintf("   within : beta=%+.3f (SE %.3f), t=%.2f, p=%.2g\n", co[2,1], co[2,2], co[2,4], co[2,5]))
  cat(sprintf("   between: beta=%+.3f (SE %.3f), t=%.2f, p=%.2g\n", co[3,1], co[3,2], co[3,4], co[3,5]))
}

cat("\n=========================================================\n")
cat(" Head-to-head:  LT ~ corrected + grid + (1|subject)  (which prediction dominates?)\n")
cat("=========================================================\n")
mboth <- lmer(LT ~ corr_z + grid_z + (1|subject), data=d, REML=FALSE)
print(round(summary(mboth)$coefficients, 3))
cat(sprintf("marginal R2 (both) = %.3f\n", marg(mboth)))

cat("\n=========================================================\n")
cat(" Within-subject dishabituation (clean of cohort): does the model predict novel>familiar?\n")
cat("=========================================================\n")
# human within-subject test-type effect, and whether the model's within prediction captures it
mh <- lmer(LT ~ trial_type + (1|subject), data=d, REML=FALSE)
ch <- summary(mh)$coefficients
cat(sprintf("HUMAN  : deviant vs background beta=%+.3f (SE %.3f), t=%.2f, p=%.2g\n", ch[2,1], ch[2,2], ch[2,4], ch[2,5]))
# model-predicted test-type effect (corrected)
agg <- aggregate(cbind(corrected_pred, grid_pred) ~ trial_type, d, mean)
cat("model mean prediction by trial_type:\n"); print(agg)
