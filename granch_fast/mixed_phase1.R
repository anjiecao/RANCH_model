# lme4 confirmation of the within-subject (deconfounded) infant analysis, Phase 1.
# For each predictor column: Mundlak split LT ~ pred_within + pred_between + (1|subject);
# within beta (per SD of prediction) and p; plus marginal R2 of LT ~ pred + (1|subject).
suppressMessages({library(lme4); library(lmerTest); library(performance)})
d <- read.csv("/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/phase1/infant_mixed_phase1.csv")
picks <- read.csv("/Users/mcfrank/Projects/ranch/RANCH_model/granch_fast/phase1/infant_mixed_phase1_picks.csv")
d$subject <- factor(d$subject)
cat(sprintf("n trials = %d, n infants = %d, cohorts = %s\n", nrow(d), nlevels(d$subject), paste(unique(d$experiment), collapse=",")))
z <- function(x) (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)
marg <- function(m) tryCatch(as.numeric(r2_nakagawa(m)$R2_marginal), error=function(e) NA)
m0 <- lmer(LT ~ 1 + (1|subject), data=d, REML=FALSE)
cat(sprintf("\n%-22s %-34s | %8s %7s %9s | %8s %7s | %7s %6s\n", "predictor", "setting", "b_within", "p_w", "b_between", "p_b", "margR2", "dAIC", "n"))
for (i in seq_len(nrow(picks))) {
  col <- picks$col[i]
  if (!(col %in% names(d))) next
  dd <- d[!is.na(d[[col]]), ]
  dd$x <- z(dd[[col]])
  gm <- ave(dd$x, dd$subject, FUN=mean)
  dd$xw <- dd$x - gm; dd$xb <- gm
  mw <- lmer(LT ~ xw + xb + (1|subject), data=dd, REML=FALSE)
  co <- summary(mw)$coefficients
  m1 <- lmer(LT ~ x + (1|subject), data=dd, REML=FALSE)
  m00 <- lmer(LT ~ 1 + (1|subject), data=dd, REML=FALSE)
  cat(sprintf("%-22s %-34s | %+8.2f %7.4f %+9.2f | %8.4f %7.3f | %7.1f %6d\n", col, picks$setting[i],
              co["xw",1], co["xw",5], co["xb",1], co["xb",5], marg(m1), AIC(m00)-AIC(m1), nrow(dd)))
}
cat("\nb_within = change in LT (s) per SD of the model prediction, using only within-infant contrasts (deconfounded);\nb_between = per SD of subject-mean prediction (confounded with cohort).\n")
