program analogweight
!27042014 Stefano Alessandrini
!compute the analog members with euclidean metrics
!input: analog_ini.txt, parameter file
!input: AnEnVarsClean.txt
!output: eps_analog.txt 
 implicit none
 
 integer:: nstep,iniziotrain,finetrain,iniziotest,finetest,leadtime,iday, itime,ivar,ltime,i1,i2,i3,i4,i5,nattempt,cont
 integer:: n_membri,numpredictor,ndaytest,ndaytrain,entrata,st,i,j,k,idaytrain,ii,iii,miss,valid,iiii,uscita
 real :: type_predictor(50),weight(50),weights(50),var_prev(50),var_prev_m1(50),var_prev_t(50),sd,std_dir,rmse1,rmse2
 real :: var_prev_tm1(50),var_prev_tp1(50),sd_var(50),error(50),var_prev_p1(50),diff_t,diff_tm1,diff_tp1,nd,mae
 real ::namepredictor(50),finish,start,rmse,rmse3,crps,brier_score,threshold,threshold_bias
 real , dimension(:,:),allocatable::eps_analog  
 real , dimension(:,:,:), allocatable :: data4d
 real , parameter :: deg2rad=0.01745329
 real,dimension(:),allocatable:: obs
 real, allocatable::weight_comb(:,:),ret(:),seq(:) !read from parameter files
 real, allocatable ::  ws(:)
    entrata=4
    uscita=5
    threshold=30 !threshold for brier score 30 kt on 0-24 hours, 55 kt on 0-48 hours, 65 kt on 0-72 iour


    open(entrata, file="analog_weight_ini.txt", status="old", iostat=st)
  
    if (st.ne.0) stop "Errore nell'apertura del file analog_weight_ini.txt"      ! nello script serve
          read(entrata,*, iostat=st)nstep,n_membri,iniziotrain,finetrain,iniziotest,finetest
          print*,nstep,n_membri,iniziotrain,finetrain,iniziotest,finetest
          read(entrata,*, iostat=st)numpredictor
          read(entrata,*, iostat=st)(type_predictor(i),i=1,numpredictor)
          read(entrata,*, iostat=st)threshold_bias
    close(entrata)
! print*,"Pesi usati",weight(1:numpredictor)
50 format(10f12.2)


 allocate (data4d((numpredictor+1),finetest,nstep))
!allocate (data4d((numpredictor+2),finetest,nstep))
allocate(obs(finetest*nstep))

ndaytest=finetest-iniziotest+1
allocate (eps_analog(ndaytest*nstep,n_membri))
ndaytrain=finetrain-iniziotrain+1
ii=0
open(entrata, file="AnEnVarsClean.txt", status="old", iostat=st)
 do j= 1,finetest
    do k= 1,nstep
        ii=ii+1
!       indice=(j-1)*flt+k
        read(entrata,*,iostat=st) (data4d(i,j,k),i=1,(numpredictor+1))
!       read(entrata,*,iostat=st) (data4d(i,j,k),i=1,(numpredictor+2))
       obs(ii)=data4d((numpredictor+1),j,k)
!        print*,"data",i,j,k,data4d(:,j,k)
      if (st > 0) stop "Errore nella lettura del file stazioni"
      if (st < 0) exit

    enddo
  enddo

print*, "Sono qui"

allocate(ws(numpredictor))


nattempt=11 ! number of possible weights

allocate(ret(numpredictor),seq(nattempt))
allocate(weight_comb(1000000,numpredictor))
ret(:)=0.
nd=1./real((nattempt-1.))
seq(1)=0.
 do i=2,nattempt
  seq(i)=seq(i-1)+nd
enddo
cont=0

call comp_weight(1,numpredictor,1,seq,ret,nattempt,cont,weight_comb)


   rmse1=1000000000.
   rmse3=1000000000.

!$omp parallel do private(weight,rmse2,eps_analog)
     do i1=1,cont
        weight(1:numpredictor)=weight_comb(i1,:)
!    if(weight(1)>0.29.and.weight(2) <0.41.and.weight(3)<0.41.and.weight(4)<0.41.and.weight(5)<0.41.and.weight(6)<0.41) then
!   if(weight(1)>0.9.and.weight(2) <0.51.and.weight(3)<0.51.and.weight(4)<0.51.and.weight(5)<0.51.and.weight(6)<0.51&
!    .and. weight(7)<0.51.and. weight(8)<0.51.and. weight(9)<0.51.and. weight(10)<0.51) then
    if(weight(1)>0.09) then
       call analog(data4d,weight,type_predictor,numpredictor,n_membri,nstep,iniziotrain,finetrain,iniziotest,&
            finetest,eps_analog,ndaytest,threshold_bias)
!      print*,"obs", obs(((iniziotest-1)*nstep+1):(iniziotest-1)*nstep+11)
!      print*,"eps",eps_analog(1:50,1)
!        rmse2=mae(obs(((iniziotest-1)*nstep+1):(finetest*nstep)),eps_analog(1:ndaytest*nstep,1:n_membri),ndaytest*nstep,n_membri)
!        rmse2=crps(obs(((iniziotest-1)*nstep+1):(finetest*nstep)),eps_analog(1:ndaytest*nstep,1:n_membri),ndaytest*nstep,n_membri)
         rmse2=rmse(obs(((iniziotest-1)*nstep+1):(finetest*nstep)),eps_analog(1:ndaytest*nstep,1:n_membri),ndaytest*nstep,n_membri)
!         rmse2=brier_score(obs(((iniziotest-1)*nstep+1):(finetest*nstep)),eps_analog(1:ndaytest*nstep,1:n_membri),&
!           ndaytest*nstep,n_membri,threshold)

         print*,"weights,rmse",weight(1:numpredictor),rmse2
!         if(weight(1)>0.99)rmse3=rmse2
         if(weight(numpredictor)<0.001.and.rmse2<rmse3)rmse3=rmse2
        if(rmse2<rmse1)then
          weights(:)=weight(:)
          rmse1=rmse2
        endif
   endif
  enddo
!$omp end parallel do
    print*,"Best weights,rmse",weights(1:numpredictor),rmse1
open(uscita, file="weight.txt")
      ! write(uscita,*)(weights(k),k=1,numpredictor),(rmse3-rmse1)/rmse3
      write(uscita,*)(weights(k),k=1,numpredictor)
     close(uscita)

end program analogweight

subroutine analog(data4d_inp,weight,type_predictor,numpredictor,n_membri,nstep,iniziotrain,finetrain,iniziotest,finetest,&
            eps_analog,ndaytest,threshold_bias)

implicit none

integer:: nstep,iniziotrain,finetrain,iniziotest,finetest,leadtime,iday, itime,ivar,ltime,ipast,ifut
integer:: n_membri,numpredictor,ndaytest,ndaytrain,entrata,st,i,j,k,idaytrain,ii,iii,miss,valid,iiii,uscita
real :: type_predictor(50),weight(50),var_prev(50),var_prev_m1(50),var_prev_t(50),sd,std_dir
real :: var_prev_tm1(50),var_prev_tp1(50),sd_var(50),error(50),var_prev_p1(50),diff_t,diff_tm1,diff_tp1
real ::namepredictor(50),dir_grad(200000),finish,start,wind_dir_error_scalar,bias,obs_bias,threshold_bias
real , dimension(ndaytest*nstep,n_membri)::eps_analog
real , dimension(finetest*nstep,3)::errvel ,errvel1,errvel2
real , dimension(finetest*nstep)::back_up
!real , dimension((numpredictor+2),finetest,nstep):: data4d,data4d_true,data4d_inp
real , dimension((numpredictor+1),finetest,nstep):: data4d,data4d_true,data4d_inp
real , parameter :: deg2rad=0.01745329

 i=0
 call cpu_time(start)
ndaytrain=finetrain-iniziotrain+1
data4d(:,:,:)=data4d_inp(:,:,:)!!!change for openmp, data4d is touched otherwise
data4d_true(:,:,:)=data4d(:,:,:)

  do iday=iniziotest,finetest
!    do iday=iniziotest,iniziotest
   do itime=1,nstep
!     do itime=nstep,nstep
     errvel(:,:)=-9999.000
    i=i+1
    data4d(:,:,:)=data4d_true(:,:,:)
    ltime=itime
     do ivar=1,numpredictor
       var_prev(ivar)= data4d(ivar,iday,ltime)
       if (var_prev(ivar)<-9998.000) then
        eps_analog(i,1:n_membri)=-9999.000  !if one of the three predicotors is missing I can't compute analog
        go to 2405
       endif
     enddo
      if(ltime /= 1)then                            !seleziono variabili previste a ltime-1
        do ivar=1,numpredictor
         var_prev_m1(ivar)=data4d(ivar,iday,ltime-1)
         if (var_prev_m1(ivar)<-9998.000) then
          eps_analog(i,1:n_membri)=-9999.000  !if one of the three predicotors is missing I can't compute analog
          go to 2405
         endif
        enddo
      endif
      if(ltime /= nstep)then                             !seleziono variabili previste a ltime+1
        do ivar=1,numpredictor
         var_prev_p1(ivar)=data4d(ivar,iday,ltime+1)
         if (var_prev_p1(ivar)<-9998.000) then
          eps_analog(i,1:n_membri)=-9999.000  !if one of the three predicotors is missing I can't compute analog
          go to 2405
         endif
        enddo
      endif

!      data4d(numpredictor+1,iday,1:nstep)=-9999.000
!      do ifut=1,50
!       if(iday<finetest-50 .and. data4d(numpredictor+2,iday+ifut,1) .eq. data4d(numpredictor+2,iday,1))&
!        data4d(numpredictor+1,iday+ifut,1:nstep)=-9999.000  ! #future measurements are not available
!      enddo
 !    if(iday<(finetest-5))then
 !     data4d(numpredictor+1,iday+1,1:nstep)=-9999.000
 !     data4d(numpredictor+1,iday+2,1:nstep)=-9999.000
 !     data4d(numpredictor+1,iday+3,1:nstep)=-9999.000
 !!     data4d(numpredictor+1,iday+4,1:nstep)=-9999.000
 !    endif
 !    do ipast=1,21
 !     if(iday>ipast .and. data4d(numpredictor+2,iday-ipast,1) .eq. data4d(numpredictor+2,iday,1) )&
 !      data4d(numpredictor+1,iday-ipast,6*ipast:nstep)=-9999.000  ! #future measurements are not available
 !    enddo
!     if(iday>2) data4d(numpredictor+1,iday-2,5:nstep)=-9999.000 ! ##
!     if(iday>3) data4d(numpredictor+1,iday-3,8:nstep)=-9999.000 ! ##
 !    if(iday>4) data4d(numpredictor+1,iday-4,11:nstep)=-9999.000 ! ##
 !    if(iday>5) data4d(numpredictor+1,iday-5,14:nstep)=-9999.000 ! ###
 !    if(iday>5) data4d(numpredictor+1,iday-5,16:nstep)=-9999.000 !
 !    print*,"ltime, varprev, varprevm1 varprevp1",ltime,var_prev(1:4),var_prev_m1(1:4),var_prev_p1(1:4)
!      if(iday==1 .and. itime==1)then
      do ivar=1,numpredictor
       if(weight(ivar)>0.00001)then
        if(type_predictor(ivar)== 1)then

!sd_var(ivar)=sd(train_altpos(,namepredictor(ivar)),na.rm=TRUE)
          sd_var(ivar)=sd(data4d(ivar,1:(finetrain),ltime),(finetrain))
          else

!  dir_grad=train_altpos(,namepredictor(ivar))
        dir_grad(1:(finetrain))=data4d(ivar,1:(finetrain),ltime)
! print*,"dirgrad",dir_grad(101:189)
        sd_var(ivar)=std_dir(dir_grad(1:(finetrain)),(finetrain))
        endif
       else
        sd_var(ivar)=1. !this is the case where a variable has weight =0, just put a value of sd without computing
       endif
      enddo
!     print*,"sdvar",sd_var(1:4)
!      endif
      ii=0

 !     do idaytrain=iniziotrain,(iday-1) 
    do idaytrain=iniziotrain,finetrain                !cycle on the training
       ii=ii+1     !indici degli errori
        do  ivar=1,numpredictor
         var_prev_t(ivar)=data4d(ivar,idaytrain,ltime)
         if (var_prev_t(ivar) < -9998.000) then
           errvel(ii,:)=-9999.000
           goto 3012 !if one of the past predictor is missing skip metric computation for that day of the training
          endif
        enddo
       if(ltime/=1)then                       !seleziono variabili previste nel passato a ltime-1
        do ivar=1,numpredictor
         var_prev_tm1(ivar)=data4d(ivar,idaytrain,(ltime-1))
         if (var_prev_tm1(ivar) < -9998.000) then
          errvel(ii,:)=-9999.000
          goto 3012 !if one of the past predictor is missing skip metric computation for that day of the training
         endif
        enddo
       endif
       if(ltime/=nstep)then                      !seleziono variabili previste nel passato a ltime+1
        do ivar=1,numpredictor
         var_prev_tp1(ivar)=data4d(ivar,idaytrain,(ltime+1))
         if (var_prev_tp1(ivar) < -9998.000) then
          errvel(ii,:)=-9999.000
          goto 3012 !if one of the past predictor is missing skip metric computation for that day of the training
         endif
        enddo
       endif
      errvel(ii,1)=0
!print*,"ltime, varprevt, varprevtm1 varprevtp1",ltime,var_prev_t(1:4),var_prev_tm1(1:4),var_prev_tp1(1:4)
      if(ltime==1)then
        do ivar=1,numpredictor
         if(weight(ivar)>0.00001)then
          if(type_predictor(ivar)==1)then !calcolo metrica differenziando se circolare o no
           diff_t=var_prev_t(ivar)-var_prev(ivar)
           diff_tp1=var_prev_tp1(ivar)-var_prev_p1(ivar)
           error(ivar)=diff_t*diff_t+diff_tp1*diff_tp1
           error(ivar)=sqrt(error(ivar))/sd_var(ivar)
         else
          diff_t=wind_dir_error_scalar(var_prev_t(ivar), var_prev(ivar))*deg2rad
          diff_tp1=wind_dir_error_scalar(var_prev_tp1(ivar), var_prev_p1(ivar))*deg2rad !          print*,"error prima",ivar,error(ivar),diff_t,diff_tp1,deg2rad
!          print*,"error prima",ivar,error(ivar),diff_t,diff_tp1,deg2rad
          error(ivar)=diff_t*diff_t+diff_tp1*diff_tp1
          error(ivar)=sqrt(error(ivar))/sd_var(ivar)
         endif
        else
         error(ivar)=0. !this is the case where a variable has weight =0, just put a value of error without computing
        endif
        errvel(ii,1)=errvel(ii,1)+error(ivar)*weight(ivar)
!         print*,"idaytrain,ivar,error(ivar)",idaytrain,ivar,error(ivar)
       enddo
      else if(ltime==nstep)then
       do ivar=1,numpredictor
        if(weight(ivar)>0.00001)then
         if(type_predictor(ivar)==1)then
          diff_t=var_prev_t(ivar)-var_prev(ivar)
          diff_tm1=var_prev_tm1(ivar)-var_prev_m1(ivar)
          error(ivar)=diff_t*diff_t+diff_tm1*diff_tm1       !calcolo metrica differenziando se circolare o n
          error(ivar)=sqrt(error(ivar))/sd_var(ivar)
         else
          diff_t=wind_dir_error_scalar(var_prev_t(ivar), var_prev(ivar))*deg2rad
          diff_tm1=wind_dir_error_scalar(var_prev_tm1(ivar), var_prev_m1(ivar))*deg2rad
          error(ivar)=diff_t*diff_t
          error(ivar)=error(ivar)+diff_tm1*diff_tm1
          error(ivar)=sqrt(error(ivar))/sd_var(ivar)
         endif
         else
          error(ivar)=0. !this is the case where a variable has weight =0, just put a value of error without computing
         endif
        errvel(ii,1)=errvel(ii,1)+error(ivar)*weight(ivar)
        enddo
       else
         do ivar=1,numpredictor
          if(weight(ivar)>0.00001)then
           if(type_predictor(ivar)==1)then
            diff_t=var_prev_t(ivar)-var_prev(ivar)
            diff_tm1=var_prev_tm1(ivar)-var_prev_m1(ivar)
            diff_tp1=var_prev_tp1(ivar)-var_prev_p1(ivar)
            error(ivar)=diff_t*diff_t+diff_tm1*diff_tm1+diff_tp1*diff_tp1
            error(ivar)=sqrt(error(ivar))/sd_var(ivar)
           else
            diff_t=wind_dir_error_scalar(var_prev_t(ivar), var_prev(ivar))*deg2rad
            diff_tm1=wind_dir_error_scalar(var_prev_tm1(ivar), var_prev_m1(ivar))*deg2rad
            diff_tp1=wind_dir_error_scalar(var_prev_tp1(ivar), var_prev_p1(ivar))*deg2rad
            error(ivar)=diff_t*diff_t+diff_tm1*diff_tm1+diff_tp1*diff_tp1
            error(ivar)=sqrt(error(ivar))/sd_var(ivar)
           endif
          else
           error(ivar)=0. !this is the case where a variable has weight =0, just put a value of error without computing
          endif
         errvel(ii,1)= errvel(ii,1)+error(ivar)*weight(ivar)
         enddo
        endif

         errvel(ii,2)=data4d(numpredictor+1,idaytrain,ltime) !associo ad ogni errore la potenza misurata alla stessa cadenza
         errvel(ii,3)=var_prev_t(1)
3012   CONTINUE
      enddo ! end cycle on the training

       errvel1(1:(ndaytrain),:)=errvel(1:(ndaytrain),:)

        miss=0
        valid=0
       do iii=1,(ndaytrain)

        if(errvel1(iii,1)<-9998.000 .or. errvel1(iii,2)<-9998.000)then
          miss=miss+1
        else
         valid=valid+1
         errvel2(valid,:)=errvel1(iii,:)
        endif

       enddo

!      call sort_matrix(errvel2,2)

      if(valid <= n_membri)then ! to deal with too many missing data
        eps_analog(i,:)=-9999.000
!        print*,"Not enough valid obs to make a complete analog forecast, all missing value are set"
       else
        back_up=errvel2(:,1)
        call   SSORT (errvel2(:,1), errvel2(:,2), valid, 2)! order the distance matrix and consequently the analogs on the second column
        errvel2(:,1)=back_up
     call   SSORT (errvel2(:,1), errvel2(:,3), valid, 2)! order the distance matrix and consequently the analogs on the second column
!       bias=obs_bias(var_prev(1),errvel2(1:n_membri,3),errvel2(1:n_membri,2),n_membri)!least square method to go from deltaf to deltaobs
        bias=obs_bias(var_prev(1),errvel2(1:valid,3),errvel2(1:valid,2),valid,threshold_bias,n_membri)!least square method to go from deltaf to deltaobs
!       bias=0.
     do iiii=1,n_membri
      
        eps_analog(i,iiii)=errvel2(iiii,2)+bias
     !  eps_analog(i,iiii)=errvel2(iiii,2)
     enddo
      endif

!    print*,"iday,itime",iday,itime
!    print*,"eps_analog",eps_analog(i,1:n_membri)

2405 CONTINUE  !!here you arrive when a missing forecast is found

    enddo !cycle itime


   enddo  !cycle iday

  call cpu_time(finish)
!  print '("Time = ",f9.3," seconds.")',finish-start

!open(uscita, file="eps_analog.txt")

!     do ii=1,(ndaytest*nstep)
!       write(uscita,*)(eps_analog(ii,k),k=1,n_membri)
!     enddo

!close(uscita)





end subroutine


real  function std_dir(dir_grad,n)
implicit none
real s1,c1,e1,b,rad2deg
real:: dir_grad(n),dir_rad(n)
integer :: count,i,n
 rad2deg=0.01745329 !pi/180
    !rad2deg=1.
    b=0.1547  ! 2/sqrt(3) - 1

      count=0
    do i=1, n
     if (dir_grad(i)>-9998.000)then
       count=count+1
       dir_rad(count)=dir_grad(i)
     endif
    enddo
   if(count==0)then
     print*,"Computing standard deviation on a variable with all missing data"
     std_dir=1.
   else
    dir_rad(1:count)=dir_rad(1:count)*rad2deg

    ! Calculate sum of sines and cosines
    s1=sum(sin(dir_rad(1:count)))/count
    c1=sum(cos(dir_rad(1:count)))/count

    !Yamartino estimator
    e1=sqrt(1.0-(s1*s1+c1*c1))

    !Standard deviation
    std_dir=asin(e1)*(1+b*e1*e1*e1)
   endif
    return
end function
real function rmse(misure,forecast,n,n_membri)
implicit none
real ::misure(n),mean
real ::forecast(n,n_membri)
integer :: count,i,n,n_membri
count=0
rmse=0
do i=1, n
!  if (var(i).ne.-9999.000)then
 if (misure(i)>-9998.000 .and.forecast(i,1) > -9998.000)then
   count=count+1
   mean=sum(forecast(i,1:n_membri))/n_membri
   rmse=rmse+ (misure(i)-mean)*(misure(i)-mean)
 !print*,"mean",mean!,!forecast(i,1:n_membri)
endif
enddo
 if(count>0)then
  rmse=rmse/count
  rmse=sqrt(rmse)
 else
  rmse=1000
 print*,"Attention no valid obs in the test period"
 endif

end function
real  function sd(var,n)
implicit none
real :: avg,variance,sum,diff
real :: var(n)
integer :: count,i,n
  count=0
  sum=0.
 do i=1, n
!  if (var(i).ne.-9999.000)then
    if (var(i)>-9998.000)then
    count=count+1
   sum=sum+var(i)
  endif
 enddo
 if(count==0)then
  print*,"Computing standard deviation on a variable with all missing data"
  sd=1.
 else
  avg=sum/count
  variance=0.
  DO i=1, n
!  if(var(i).ne.-9999.000)then
   if(var(i)>-9998.000)then
    diff=var(i)-avg
    variance=variance+(diff*diff)/(count-1)
   endif
  END DO
  sd=variance**0.5
 endif
return
end function

subroutine sort_matrix( matrix, col )
real, dimension(:,:) :: matrix
integer :: col
real  :: store_column
!
! Work arrays
!
integer, dimension(size(matrix,1)) :: order

!
! Your favourite sorting algorithm - here bubblesort
!
order = (/ (i, i=1,size(matrix,1)) /)
!
! Sort the array order, using the right matrix column
!
 do j = 1,size(matrix,1)
  do i = j+1,size(matrix,1)
   if ( matrix(col,order(i)) < matrix(col,order(j)) ) then
   k = order(i)
   order(i) = order(j)
   order(j) = k
   endif
  enddo
 enddo
!
! Reorder the columns via new_matrix
!
 do j = 1,size(matrix,1)
  store_column = matrix(i,j)
  matrix(j,:) = matrix(order(j),:)
  matrix(order(j),:) = store_column
 enddo

end subroutine
SUBROUTINE SSORT (X, Y, N, KFLAG)
!***BEGIN PROLOGUE  SSORT
!***PURPOSE  Sort an array and optionally make the same interchanges in
!            an auxiliary array.  The array may be sorted in increasing
!            or decreasing order.  A slightly modified QUICKSORT
!            algorithm is used.
!***LIBRARY   SLATEC
!***CATEGORY  N6A2B
!***TYPE      SINGLE PRECISION (SSORT-S, DSORT-D, ISORT-I)
!***KEYWORDS  SINGLETON QUICKSORT, SORT, SORTING
!***AUTHOR  Jones, R. E., (SNLA)
!           Wisniewski, J. A., (SNLA)
!***DESCRIPTION
!
!   SSORT sorts array X and optionally makes the same interchanges in
!   array Y.  The array X may be sorted in increasing order or
!   decreasing order.  A slightly modified quicksort algorithm is used.
!
!   Description of Parameters
!      X - array of values to be sorted   (usually abscissas)
!      Y - array to be (optionally) carried along
!      N - number of values in array X to be sorted
!      KFLAG - control parameter
!            =  2  means sort X in increasing order and carry Y along.
!            =  1  means sort X in increasing order (ignoring Y)
!            = -1  means sort X in decreasing order (ignoring Y)
!            = -2  means sort X in decreasing order and carry Y along.
!
!***REFERENCES  R. C. Singleton, Algorithm 347, An efficient algorithm
!               for sorting with minimal storage, Communications of
!               the ACM, 12, 3 (1969), pp. 185-187.
!  761101  DATE WRITTEN
!  761118  Modified to use the Singleton quicksort algorithm.  (JAW)
!  890531  Changed all specifi!intrinsics to generic.  (WRB)
!  890831  Modified array declarations.  (WRB)
!  891009  Removed unreferenced statement labels.  (WRB)
!  891024  Changed category.  (WRB)
!  891024  REVISION DATE from Version 3.2
!  891214  Prologue converted to Version 4.0 format.  (BAB)
!  900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
!  901012  Declared all variables; changed X,Y to SX,SY. (M. McClain)
!  920501  Reformatted the REFERENCES section.  (DWL, WRB)
!  920519  Clarified error messages.  (DWL)
!  920801  Declarations section rebuilt and code restructured to use
!          IF-THEN-ELSE-ENDIF.  (RWC, WRB)
!C***END PROLOGUE  SSORT
!    .. Scalar Arguments ..
INTEGER KFLAG, N
!    .. Array Arguments ..
REAL X(*), Y(*)
!    .. Local Scalars ..
REAL R, T, TT, TTY, TY
INTEGER I, IJ, J, K, KK, L, M, NN
!    .. Local Arrays ..
INTEGER IL(21), IU(21)
!    .. External Subroutines ..
!    None
!    .. Intrinsi!Functions ..
INTRINSIC ABS, INT
!C***FIRST EXECUTABLE STATEMENT  SSORT
NN = N
IF (NN .LT. 1) THEN
PRINT *,      'The number of values to be sorted is not positive.'
RETURN
ENDIF
!C
KK = ABS(KFLAG)
IF (KK.NE.1 .AND. KK.NE.2) THEN
PRINT *,     'The sort control parameter, K, is not 2, 1, -1, or -2.'
RETURN
ENDIF
!C
!    Alter array X to get decreasing order if needed
!C
IF (KFLAG .LE. -1) THEN
DO 10 I=1,NN
X(I) = -X(I)
10    CONTINUE
ENDIF
!C
IF (KK .EQ. 2) GO TO 100
!C
!    Sort X only
!C
M = 1
I = 1
J = NN
R = 0.375E0
!C
20 IF (I .EQ. J) GO TO 60
IF (R .LE. 0.5898437E0) THEN
R = R+3.90625E-2
ELSE
R = R-0.21875E0
ENDIF
!C
30 K = I
!C
!    Select a central element of the array and save it in location T
!C
IJ = I + INT((J-I)*R)
T = X(IJ)
!C
!    If first element of array is greater than T, interchange with T
!C
IF (X(I) .GT. T) THEN
X(IJ) = X(I)
X(I) = T
T = X(IJ)
ENDIF
L = J
!C
!    If last element of array is less than than T, interchange with T
!C
IF (X(J) .LT. T) THEN
X(IJ) = X(J)
X(J) = T
T = X(IJ)
!C
!       If first element of array is greater than T, interchange with T
!C
IF (X(I) .GT. T) THEN
X(IJ) = X(I)
X(I) = T
T = X(IJ)
ENDIF
ENDIF
!C
!    Find an element in the second half of the array which is smaller
!    than T
!C
40 L = L-1
IF (X(L) .GT. T) GO TO 40
!C
!    Find an element in the first half of the array which is greater
!    than T
!C
50 K = K+1
IF (X(K) .LT. T) GO TO 50
!C
!    Interchange these elements
!C
IF (K .LE. L) THEN
TT = X(L)
X(L) = X(K)
X(K) = TT
GO TO 40
ENDIF
!C
!    Save upper and lower subscripts of the array yet to be sorted
!C
IF (L-I .GT. J-K) THEN
IL(M) = I
IU(M) = L
I = K
M = M+1
ELSE
IL(M) = K
IU(M) = J
J = L
M = M+1
ENDIF
GO TO 70
!C
!    Begin again on another portion of the unsorted array
!C
60 M = M-1
IF (M .EQ. 0) GO TO 190
I = IL(M)
J = IU(M)
!C
70 IF (J-I .GE. 1) GO TO 30
IF (I .EQ. 1) GO TO 20
I = I-1
!C
80 I = I+1
IF (I .EQ. J) GO TO 60
T = X(I+1)
IF (X(I) .LE. T) GO TO 80
K = I
!C
90 X(K+1) = X(K)
K = K-1
IF (T .LT. X(K)) GO TO 90
X(K+1) = T
GO TO 80
!C
!    Sort X and carry Y along
!C
100 M = 1
I = 1
J = NN
R = 0.375E0
!C
110 IF (I .EQ. J) GO TO 150
IF (R .LE. 0.5898437E0) THEN
R = R+3.90625E-2
ELSE
R = R-0.21875E0
ENDIF
!C
120 K = I
!C
!    Select a central element of the array and save it in location T
!C
IJ = I + INT((J-I)*R)
T = X(IJ)
TY = Y(IJ)
!C
!    If first element of array is greater than T, interchange with T
!C
IF (X(I) .GT. T) THEN
X(IJ) = X(I)
X(I) = T
T = X(IJ)
Y(IJ) = Y(I)
Y(I) = TY
TY = Y(IJ)
ENDIF
L = J
!C
!    If last element of array is less than T, interchange with T
!C
IF (X(J) .LT. T) THEN
X(IJ) = X(J)
X(J) = T
T = X(IJ)
Y(IJ) = Y(J)
Y(J) = TY
TY = Y(IJ)
!C
!       If first element of array is greater than T, interchange with T
!C
IF (X(I) .GT. T) THEN
X(IJ) = X(I)
X(I) = T
T = X(IJ)
Y(IJ) = Y(I)
Y(I) = TY
TY = Y(IJ)
ENDIF
ENDIF
!C
!    Find an element in the second half of the array which is smaller
!    than T
!C
130 L = L-1
IF (X(L) .GT. T) GO TO 130
!C
!    Find an element in the first half of the array which is greater
!    than T
!C
140 K = K+1
IF (X(K) .LT. T) GO TO 140
!C
!    Interchange these elements
!C
IF (K .LE. L) THEN
TT = X(L)
X(L) = X(K)
X(K) = TT
TTY = Y(L)
Y(L) = Y(K)
Y(K) = TTY
GO TO 130
ENDIF
!C
!    Save upper and lower subscripts of the array yet to be sorted
!C
IF (L-I .GT. J-K) THEN
IL(M) = I
IU(M) = L
I = K
M = M+1
ELSE
IL(M) = K
IU(M) = J
J = L
M = M+1
ENDIF
GO TO 160
!C
!    Begin again on another portion of the unsorted array
!C
150 M = M-1
IF (M .EQ. 0) GO TO 190
I = IL(M)
J = IU(M)
!C
160 IF (J-I .GE. 1) GO TO 120
IF (I .EQ. 1) GO TO 110
I = I-1
!C
170 I = I+1
IF (I .EQ. J) GO TO 150
T = X(I+1)
TY = Y(I+1)
IF (X(I) .LE. T) GO TO 170
K = I
!C
180 X(K+1) = X(K)
Y(K+1) = Y(K)
K = K-1
IF (T .LT. X(K)) GO TO 180
X(K+1) = T
Y(K+1) = TY
GO TO 170
!C
!    Clean up
!C
190 IF (KFLAG .LE. -1) THEN
DO 200 I=1,NN
X(I) = -X(I)
200    CONTINUE
ENDIF
RETURN
END

real function wind_dir_error_scalar (dir1, dir2)
implicit none

real :: dir1, dir2	! input wind directions to compare
real :: output	! result wind direction, degrees 0-360

real :: sol1, sol2			! local variables

sol1   = abs (dir1 - dir2)		! minimum absolute angular separation
sol2   = abs (sol1 - 360)		! on the 360 degree circle
wind_dir_error_scalar = min (sol1, sol2)
return
end
recursive subroutine comp_weight(par,n,pos,seq,ret,m,cont,weight)
integer n,m,cont
real,intent(out):: ret(n),weight(1000000,n) !ste only 10^6 possible combinationis allowed, increased in case more comb. wanted
real,intent(in)::   seq(m)
integer,intent(in):: par,pos
ret(par)=seq(pos)
if(nint(sum(ret)*1000)==1000 .and. par==n)then
cont=cont+1
weight(cont,:)=ret
endif
if(par <n)call comp_weight(par+1,n,1,seq,ret,m,cont,weight)
!only 11 weights from 0 to 1 are allowed
if(pos<m)call comp_weight(par,n,pos+1,seq,ret,m,cont,weight)

end subroutine comp_weight
real function mae(misure,forecast,n,n_membri)
implicit none
real ::misure(n),mean,finto(n_membri)
real ::forecast(n,n_membri)
integer :: count,i,n,n_membri
count=0
mae=0
do i=1, n
!  if (var(i).ne.-9999.000)then
if (misure(i)>-9998.000 .and.forecast(i,1) > -9998.000)then
count=count+1
!mean=sum(forecast(i,1:n_membri))/n_membri
!print*,"membri",forecast(i,1:20)
call SSORT(forecast(i,1:n_membri),finto, n_membri, 1)
!print*,"membri dopo",forecast(i,1:20)
mean=sum(forecast(i,10:11))/2. !!!median of the distribution with 20 sorted members
mae=mae+ abs(misure(i)-mean)
!print*,"mean",mean!,!forecast(i,1:n_membri)
endif
enddo
if(count>0)then
mae=mae/count

else
mae=1000
print*,"Attention no valid obs in the test period"
endif

end function
real function crps(misure,forecast,n,n_membri)
implicit none
real ::misure(n),mean,score,scorel,scorer
real ::forecast(n,n_membri)
integer :: count,i,n,n_membri,j

count=0
crps=0
do i=1, n
!  if (var(i).ne.-9999.000)then
if (misure(i)>-9998.000 .and.forecast(i,1) > -9998.000)then
count=count+1
scorel=sum(abs(forecast(i,:)-misure(i)))/n_membri
scorer=0.
do j=1,n_membri
scorer=scorer+sum(abs(forecast(i,j)-forecast(i,:)))
enddo
scorer=scorer/(2*n_membri*n_membri)
crps=crps+(scorel-scorer)
!print*,"mean",mean!,!forecast(i,1:n_membri)
endif
enddo

if(count>0)then
crps =crps/count
else
crps=1000
print*,"Attention no valid obs in the test period"
endif

end function
real  function obs_bias(cur_fc,past_fc,past_obs,n_data,threshold_bias,n_members)
implicit none
real cur_fc,m,q,ymean,xmean,deltaf,threshold_bias,xmean_mem
real past_fc(n_members),past_obs(n_members)
integer n_members,n_data

ymean=sum(past_obs(:))/n_data
xmean=sum(past_fc(:))/n_data
xmean_mem=sum(past_fc(:))/n_members
m=sum((past_fc(:)-xmean)*(past_obs(:)-ymean))/(sum((past_fc(:)-xmean)**2))
q=ymean-m*xmean! intercept
deltaf=cur_fc-xmean_mem
if(cur_fc<threshold_bias)then
 obs_bias=0.
else
 obs_bias=m*deltaf
endif
! obs_bias=deltaf*0.
!print*,cur_fc,xmean,ymean,obs_bias
end function

real function brier_score(misure,forecast,n,n_membri,threshold)
implicit none
real ::misure(n),score,scorel,scorer,threshold
real ::forecast(n,n_membri)
integer :: count,i,n,n_membri,j

count=0
brier_score=0.
do i=1, n
!  if (var(i).ne.-9999.000)then
 if (misure(i)>-9998.000 .and.forecast(i,1) > -9998.000)then
  count=count+1
  scorel=0.
  if(misure(i)>threshold)scorel=1.
  scorer=0
  do j=1,n_membri
   if(forecast(i,j)>threshold)scorer=scorer+1 ! scorer= number of memebers that are greter than threshold
  enddo
  brier_score=brier_score+((scorer/n_membri)-scorel)*((scorer/n_membri)-scorel)

 endif
enddo

if(count>0)then
 brier_score=brier_score/count
else
  brier_score=1000
  print*,"Attention no valid obs in the test period"
endif

end function
