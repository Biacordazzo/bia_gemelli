library(fdapace)
library(rTensor)
library(KernSmooth)
library(locfit)
library(vegan)
library(pROC)


#' Temporal Functional SVD
#' @param datlist A list of n matrices data, each matrix represents a subject;
#' the first row represents the sampling timestamp; 
#' following rows represent the feature measurements.
#' @param r CP-rank/Number of principle components. Default: 3.
#' @param interval The range of time points. Default: range of all observed data. 
#' @param resolution Resolution for the output eigencurves. Default: 100.
#' @param smooth Smoothing parameter for RKHS norm. Larger means smoother functions. Default: 1e-8.
#' @param maxiter Maximum number of itereation. Default: 20.
#' @param epsilon Convergence criteria for a and b. Default: 1e-4.
#' @return The estimations of the principle components in suject/feature/time;
#'       Var.prop: Explained variances.
ftsvd <- function(datlist, interval = NULL, r = 3, resolution = 251, smooth=1e-8,
                  maxiter=20, epsilon=1e-4){
  n = length(datlist)
  p = nrow(datlist[[1]])-1
  
  Lambda = rep(0, r)
  A = matrix(0, n, r)
  B = matrix(0, p, r)
  Phi = matrix(0, resolution, r)
  PCname <- paste('Component', 1:r)
  colnames(A) = PCname
  colnames(B) = PCname
  colnames(Phi) = PCname
  rownames(A) = names(datlist)
  rownames(B) = rownames(datlist[[1]])[-1]
  
  
  # Calculate range.
  timestamps.all = NULL
  for (i in 1:n){
    timestamps.all = c(timestamps.all, datlist[[i]][1,])
  }
  
  timestamps.all = sort(unique(timestamps.all))
  if (is.null(interval)){
    interval = c(timestamps.all[1], timestamps.all[length(timestamps.all)])
  }
  
  # rescale the time to 0-1.
  input.time.range = c(timestamps.all[1], timestamps.all[length(timestamps.all)])
  for (i in 1:n){
    datlist[[i]][1,] = (datlist[[i]][1,] - input.time.range[1]) / (input.time.range[2] - input.time.range[1])
  }
  interval = (interval - input.time.range[1]) / (input.time.range[2] - input.time.range[1])
  
  res = NULL
  Lambda = rep(0, r)
  X = NULL
  y = NULL
  
  ti <- vector(mode='list', length=n)
  for (i in 1:n){
    temp = 1 + round((resolution-1) * (datlist[[i]][1,] - interval[1]) / (interval[2] - interval[1]))
    temp[which(temp<=0 | temp>resolution)] = 0
    ti[[i]] <- temp
  }
  
  tipos <- vector(mode='list', length=n)
  for (i in 1:n){
    keep <- ti[[i]]>0
    y <- c(y, as.vector(t(datlist[[i]][-1,keep])))
    tipos[[i]] <- keep
  }
  
  for (s in 1:r){ 
    # calculate rank-1 component sequentially.
    # Step 1: initialization.
    print(sprintf("Calculate the %dth Component", s))
    
    # b.hat = b.initials[,s]
    # obtain intialization of b
    data.unfold = NULL
    for (i in 1:n){
      data.unfold = cbind(data.unfold, datlist[[i]][2:(p+1),])
    }
    b.initials <- svd(data.unfold, nu=r, nv=r)$u
    b.hat = b.initials[,1]
    
    # compress the data and apply function PCA.
    Lt = list()
    Ly = list()
    for (i in 1:n){
      Lt = c(Lt, list(datlist[[i]][1,]))
      Ly = c(Ly, list(as.numeric(b.hat%*%datlist[[i]][2:(p+1),])))
    }
    phi.hat = freg_rkhs(Lt,Ly,rep(1,length(Lt))/sqrt(length(Lt)),interval=interval, 
                        resolution = resolution, smooth=smooth)
    phi.hat = phi.hat / sqrt(sum(phi.hat^2))
    
    # iteratively update a,b,phi
    a.hat <- rep(1,n)
    t <- 0
    dif <- 1
    while(t<=maxiter & dif>epsilon){
      # update a:
      a.tilde <- rep(0,n)
      for (i in 1:n){
        t.temp <- tipos[[i]]
        a.tilde[i] <- b.hat %*% datlist[[i]][2:(p+1),t.temp] %*% phi.hat[ti[[i]][t.temp]] 
        a.tilde[i] <- a.tilde[i] / sum((phi.hat[ti[[i]][t.temp]])^2)
      }
      a.new <- a.tilde / sqrt(sum(a.tilde^2))
      dif <- sum((a.hat - a.new)^2)
      a.hat <- a.new
      
      # update b
      temp.num <- matrix(0,p,n)
      temp.denom <- rep(0,n)
      for (i in 1:n){
        t.temp <- tipos[[i]]
        temp.num[,i] <- datlist[[i]][2:(p+1),t.temp] %*% phi.hat[ti[[i]][t.temp]]
        temp.denom[i] <-sum((phi.hat[ti[[i]][t.temp]])^2)
      }
      b.tilde <- as.numeric(temp.num%*%a.hat) / as.numeric(temp.denom%*%(a.hat^2))
      b.new <- b.tilde / sqrt(sum(b.tilde^2))
      dif <- max(dif, sum((b.hat - b.new)^2))
      b.hat <- b.new
      
      # updata phi:
      Lt = list()
      Ly = list()
      for (i in 1:n){
        Lt = c(Lt, list(datlist[[i]][1,]))
        Ly = c(Ly, list(as.numeric(b.hat%*%datlist[[i]][2:(p+1),])))
      }
      
      # updata phi:
      phi.hat = freg_rkhs(Lt,Ly,a.hat,interval=interval,
                          resolution = resolution, smooth=smooth)
      phi.hat = phi.hat / sqrt(sum(phi.hat^2))
      t <- t+1
    }
    
    # calculate lambda
    x = NULL
    for (i in 1:n){
      t.temp = ti[[i]]
      t.temp <- t.temp[t.temp>0]
      x <- c(x,as.vector(t(a.hat[i]*b.hat%o%phi.hat[t.temp])))
    }
    X = cbind(X, x)
    l.fit = lm(y~x-1)
    lambda = as.numeric(l.fit$coefficients)
    A[,s] = a.hat
    #P.A = P.A - a.hat %*% t(a.hat)
    B[,s] = b.hat
    Phi[,s] = t(phi.hat)
    Lambda[s] = lambda
    
    # update datlist
    for (i in 1:n){
      temp <- tipos[[i]]
      datlist[[i]][2:(p+1),which(temp)] = datlist[[i]][2:(p+1),which(temp)] - 
        Lambda[s] * A[i,s] * (B[,s] %*% t(Phi[ti[[i]][temp],s])) 
    }
    print(paste0("Convergence reached at dif=", dif, ', iter=', t))
  }
  l.fit = lm(y~X-1)
  Lambda = as.numeric(l.fit$coefficients)
  
  # revise the sign of Lambda
  for (r in length(Lambda)){
    if (Lambda[r]<0){
      Lambda[r] = -Lambda[r]
      B[,r] = -B[,r]
    }
  }
  
  time.return = seq(interval[1],interval[2],length.out = resolution)
  time.return = time.return * (input.time.range[2] - input.time.range[1]) + input.time.range[1]
  results = list("A.hat" = A, "B.hat" = B, 
                 "Phi.hat" = Phi, "time" = time.return,
                 "Lambda" = Lambda, "r.square" = summary(l.fit)$r.square)
  return(results)
}



log_comp_centralize <- function(datlist, r = 3){
  n = length(datlist)
  p = nrow(datlist[[1]])-1
  log.comp = matrix(0,n,p)
  for (i in 1:length(datlist)){
    log.comp[i,] = apply(datlist[[i]][-1,], 1, mean)
  }
  log.comp.svd = svd(log.comp, nu=r, nv=r)
  log.comp.mean = log.comp.svd$u %*% t(log.comp.svd$v * log.comp.svd$d[1:r])
  mf.new = datlist
  for (i in 1:length(datlist)){
    mf.new[[i]][-1,] = datlist[[i]][-1,] - log.comp.mean[i,]
  }
  results = list("datlist" = mf.new, "A.tilde" = log.comp.svd$u, 
                 "B.tilde" = log.comp.svd$v, "lambda.tilde" = log.comp.svd$d[1:r])
  return(results)
}


#' Format data table into input of ftsvd
#' @param taxon_table A table of read counts, with n rows for samples and p columns for taxa.
#' @param time_point The time stamp of each sample, relative to the start of the study. 
#' A length n vector.
#' @param subjectID The subject ID of each sample. A length n vector.
#' @param threshold A threshold for taxon filtering. 
#' Taxa with zero counts percentage >= threshold will be excluded.
#' @param pseudo_count A small number to add to all the counts before 
#' normalizing into proportions and log transformation.
#' @return Input for ftsvd. A list of matrices, each representing the 
format_ftsvd <- function(taxon_table, time_point, subjectID, threshold=0.8, feature_names=NULL, pseudo_count=0.5, logtransform=T){
  # format data table into a list as input for ftsvd(), 
  # read counts all have 1/2 added, before being normalized into proportions and log transformation
  # check length of subID and time_point
  if (length(subjectID)!=nrow(taxon_table)) 
    stop('length of subjectID does not match taxon_table!')
  if (length(time_point)!=nrow(taxon_table)) 
    stop('length of time_point does not match taxon_table!')
  # keep taxon that has non-zeros in >=1-threshold samples
  if (is.null(feature_names)){
    taxon_table <- taxon_table[,colMeans(taxon_table==0)<threshold]
  }else{
    taxon_table <- taxon_table[,feature_names]
  }
  taxon_table <- taxon_table+pseudo_count
  if(logtransform){
    taxon_table <- t(log(taxon_table/rowSums(taxon_table)))
  }else{
    taxon_table <- t(taxon_table)
  }
  taxon_table <- rbind(time_point, taxon_table)
  rownames(taxon_table)[1] <- 'time_point'
  subID <- unique(subjectID)
  nsub <- length(subID)
  
  # construct list of data matrices, each element representing one subject
  datlist <- vector("list", length = nsub)
  names(datlist) <- subID
  
  # Each slice represents an individual (unequal sized matrix).
  for (i in 1:nsub){
    # print(i)
    datlist[[i]] <- taxon_table[, subjectID==subID[i]]
    datlist[[i]] <- datlist[[i]][,order(datlist[[i]][1,])]
    datlist[[i]] <- datlist[[i]][,!duplicated(datlist[[i]][1,])]
    colnames(datlist[[i]]) <- NULL
  }
  return(datlist)
}




##########################
# RKHS functional regression
#########################
bernoulli_kernel <- function(x, y){
  k1.x = x-0.5
  k1.y = y-0.5
  k2.x = 0.5*(k1.x^2-1/12)
  k2.y = 0.5*(k1.y^2-1/12)
  xy = abs(x %*% t(rep(1,length(y))) - rep(1,length(x)) %*% t(y))
  k4.xy = 1/24 * ((xy-0.5)^4 - 0.5*(xy-0.5)^2 + 7/240)
  kern.xy = k1.x %*% t(k1.y) + k2.x %*% t(k2.y) - k4.xy + 1
  return(kern.xy)
}

freg_rkhs <- function(Lt, Ly, z, interval, resolution = 101, smooth=1e-8){
  tm = NULL
  for (i in 1:length(Lt)) {
    tm = c(tm, Lt[[i]])
  }
  A = matrix(0, length(tm), length(tm))
  A2 = NULL
  c = rep(0, length(tm))
  for(i in 1:length(Lt)){
    Ki = bernoulli_kernel(Lt[[i]], tm)
    A = A + z[i]^2 * t(Ki)%*%Ki
    A2 = rbind(A2, Ki)
    c = c + z[i] * t(Ki) %*% Ly[[i]]
  }
  A.temp = A + smooth * A2
  A.temp.eig = eigen(A.temp, symmetric = TRUE)
  A.d = A.temp.eig$value
  A.d[A.d<1e-10] = 1e-10
  
  beta = solve( (A.temp.eig$vector)%*%(t(A.temp.eig$vector)*A.d) ) %*% c
  
  phi.est = bernoulli_kernel(seq(interval[1],interval[2],length.out = resolution), tm) %*% beta
  #plot(phi.est)
  return(phi.est)
}


#' Multiply loadings from res_ftsvd into denoised tensor
#' @param res_ftsvd Output of ftsvd
#' @param mean_svd Outpuf of log_comp_centralize
#' @return The denoised functional tensor
tdenoise <- function(res_ftsvd, mean_svd=NULL){
  n <- nrow(res_ftsvd$A.hat)
  p <- nrow(res_ftsvd$B.hat)
  resol <- nrow(res_ftsvd$Phi.hat)
  tensor.est <- array(0,dim=c(n,p,resol))
  if (!is.null(mean_svd))
    tensor.est <- (mean_svd$A.tilde %*% t(mean_svd$B.tilde * mean_svd$lambda.tilde)) %o%
    rep(1, resol)
  for (i in 1:ncol(res_ftsvd$A.hat)){
    tensor.est <- tensor.est+res_ftsvd$A.hat[,i]%o%res_ftsvd$B.hat[,i]%o%res_ftsvd$Phi.hat[,i]*res_ftsvd$Lambda[i]
  }
  dimnames(tensor.est)[[3]] <- res_ftsvd$time
  return(tensor.est)
}




plot_time_loading <- function(res){
  Phi.data <- res$Phi.hat
  npc <- ncol(Phi.data)
  ntime <- nrow(Phi.data)
  Phi.data <- data.frame(time=res$time, value=as.vector(Phi.data), 
                         component=as.factor(as.vector(t(matrix(rep(1:npc,ntime),npc,)))))
  ptime <- ggplot(data=Phi.data, aes(x=time, y=value, color=component)) + geom_line()
  return(ptime)
}

aggregate_feature <- function(res_ftsvd, res_svd, datlist){
  # estimated
  B.data <- as.data.frame(res_ftsvd$B.hat)
  tensor.est <- tdenoise(res_ftsvd, res_svd)
  tensor.est.agg <- apply(tensor.est, c(1,3), function(x){(t(B.data)%*%x)})
  dim(tensor.est.agg)
  npc <- ncol(B.data)
  metafeature.est <- NULL
  for (i in 1:npc){ 
    tmp <- data.frame(value=as.vector(tensor.est.agg[i,,]),
                      subID=rep(dimnames(tensor.est.agg)[[2]], dim(tensor.est.agg)[3]),
                      timepoint=as.vector(t(matrix(res_ftsvd$time,length(res_ftsvd$time),dim(tensor.est.agg)[2]))),
                      PC=paste0('Component ',i))
    metafeature.est <- rbind(metafeature.est,tmp)
  }
  metafeature.est$type <- 'estimated'
  # observed
  datlist.agg <- sapply(datlist, function(x){t(B.data)%*%x[-1,]}, simplify=F)
  metafeature.obs <- NULL
  for (i in 1:length(datlist.agg)){
    tmp <- data.frame(value=as.vector(datlist.agg[[i]]), 
                      subID=names(datlist.agg)[i],
                      timepoint=as.vector(t(matrix(datlist[[i]][1,], ncol(datlist[[i]]), npc))),
                      PC=rep(rownames(datlist.agg[[i]]), ncol(datlist.agg[[i]])))
    
    metafeature.obs <- rbind(metafeature.obs, tmp)
  }
  metafeature.obs$type <- 'observed'
  return(list(metafeature.obs=metafeature.obs, 
              metafeature.est=metafeature.est))
}


aggregate_subject <- function(res_ftsvd, res_svd, datlist){
  # estimated
  A.data <- as.data.frame(res_ftsvd$A.hat)
  tensor.est <- tdenoise(res_ftsvd, res_svd)
  tensor.est.agg <- apply(tensor.est, c(2,3), function(x){(t(A.data)%*%x)})
  dim(tensor.est.agg)
  npc <- ncol(A.data)
  metasubject.est <- NULL
  for (i in 1:npc){ 
    tmp <- data.frame(value=as.vector(tensor.est.agg[i,,]),
                      feature=rep(dimnames(tensor.est.agg)[[2]], dim(tensor.est.agg)[3]),
                      timepoint=as.vector(t(matrix(res_ftsvd$time,length(res_ftsvd$time),dim(tensor.est.agg)[2]))),
                      PC=paste0('Component ',i))
    metasubject.est <- rbind(metasubject.est,tmp)
  }
  
  return(metasubject.est)
}


tabular_feature_est <- function(tensor.est, feature_sel){
  tensor.est <- tensor.est[,feature_sel,]
  tab_est <- NULL
  tm <- as.numeric(dimnames(tensor.est)[[3]])
  for (i in 1:dim(tensor.est)[1]){
    tmp <- data.frame(subID=rownames(tensor.est)[i],
                      time=as.vector(t(matrix(tm, length(tm), ncol(tensor.est)))),
                      feature=rep(colnames(tensor.est), length(time)),
                      value=as.vector(tensor.est[i,,]))
    tab_est <- rbind(tab_est, tmp)
  }
  tab_est$type <- 'estimated'
  return(tab_est)
}


tabular_feature_obs <- function(datlist, feature_sel){
  tab_obs <- NULL
  for (i in 1:length(feature_sel)){
    value <- unlist(sapply(datlist, function(x){x[feature_sel[i],]}, simplify=F))
    time_point <- unlist(sapply(datlist, function(x){x['time_point',]}, simplify=F))
    nobs <- sapply(datlist, function(x){ncol(x)}, simplify=F)
    subID <- unlist(mapply(function(i){rep(names(datlist)[i], nobs[i])}, 
                           1:length(nobs), SIMPLIFY=F))
    tmp <- data.frame(subID=subID, time=time_point, feature=feature_sel[i], value=value)
    rownames(tmp) <- NULL
    tab_obs <- rbind(tab_obs, tmp)
  }
  tab_obs$type <- 'observed'
  return(tab_obs)
}



#' Calculate ROC using logistic regression
#' @param clust the observed labels
#' @param Xdata the predictors
#' @return output of function roc 
roc_logistic <- function(xtrain, ytrain, xtest, ytest){
  dftrain <- data.frame(y=ytrain, x=xtrain)
  dftest <- data.frame(y=ytest, x=xtest)
  fit <- glm(y ~ ., data = dftrain, family = "binomial")
  prob <- predict(fit, newdata=dftest, type = c("response"))
  g <- roc(dftest$y~prob)
  return(g$auc)
}

#' Estimate A of testing data based on B and Phi from training data
#' @param datlist testing data
#' @param res_ftsvd ftsvd result from training data
#' @return estimated A of testing data
est_A <- function(datlist, res_ftsvd){
  B <- res_ftsvd$B.hat
  Phi <- res_ftsvd$Phi.hat
  Lambda <- res_ftsvd$Lambda
  time.return <- res_ftsvd$time
  n <- length(datlist)
  p <- nrow(B)
  r <- ncol(B)
  resolution <- length(time.return)
  A_test <- matrix(0,n,r)
  y <- NULL
  
  ti <- vector(mode = "list", length = n)
  for (i in 1:n){
    ti[[i]] <- sapply(datlist[[i]][1,], function(x){which.min(abs(x-time.return))})
  }
  
  for (i in 1:n){
    for (j in 1:p){
      y.temp = as.numeric(datlist[[i]][j+1,])
      y = c(y, y.temp[ti[[i]]>0])
    }
  }
  
  for (s in 1:r){
    for (i in 1:n){
      yi <- as.numeric(B[,s]%*%datlist[[i]][2:(p+1),])
      t.temp <- ti[[i]]
      A_test[i,s] <- sum(yi[t.temp>0] * Phi[t.temp[t.temp>0],s]) / sum(t.temp>0)
    }
    A_test[,s]  <- A_test[,s] / sqrt(sum(A_test[,s]^2))
    
    # calculate lambda
    x = NULL
    for (i in 1:n){
      t.temp = ti[[i]]
      for (j in 1:p){
        x = c(x, A_test[i,s]*B[j,s]*Phi[t.temp[t.temp>0],s])
      }
    }
    l.fit = lm(y~x-1)
    Lambda[s] = as.numeric(l.fit$coefficients)
    
    for (i in 1:n){
      temp <- ti[[i]]
      datlist[[i]][2:(p+1),which(temp>0)] <- datlist[[i]][2:(p+1),which(temp>0)] - 
        Lambda[s] * A_test[i,s] * (B[,s] %*% t(Phi[temp[temp>0],s])) 
    }
  }
  rownames(A_test) <- names(datlist)
  colnames(A_test) <- paste('Component', 1:r)
  return(A_test)
}

