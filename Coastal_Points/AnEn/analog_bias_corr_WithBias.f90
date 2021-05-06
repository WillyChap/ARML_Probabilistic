 program analog
!27042014 Stefano Alessandrini
!compute the analog members with euclidean metrics
!input: analog_ini.txt, parameter file
!input: AnEnVarsClean.txt
!output: eps_analog.txt 
 implicit none
 
 integer:: nstep,iniziotrain,finetrain,iniziotest,finetest,leadtime,iday, itime,ivar,ltime,ipast
 integer:: n_membri,numpredictor,ndaytest,ndaytrain,entrata,st,i,j,k,idaytrain,ii,iii,miss,valid,iiii,uscita,uscita1,uscita3
 real :: type_predictor(50),weight(50),var_prev(50),var_prev_m1(50),var_prev_t(50),sd,std_dir,wind_dir_error_scalar
 real :: var_prev_tm1(50),var_prev_tp1(50),sd_var(50),error(50),var_prev_p1(50),diff_t,diff_tm1,diff_tp1
 real ::namepredictor(50),finish,start,bias,obs_bias,sig_bias,mean_an,sigma_bias,threshold_bias
 real , dimension(:,:),allocatable::errvel,eps_analog,errvel1,errvel2
 REAL, dimension(:,:),allocatable::num_hurricane,date_hurricane
Real, dimension(:),allocatable::back_up,dir_grad
 real , dimension(:,:,:), allocatable :: data4d,data4d_true
 real , parameter :: deg2rad=0.01745329

 !read from parameter files
    open(entrata, file="analog_ini.txt", status="old", iostat=st)
    print*,"Sono qui",st
    if (st.ne.0) stop "Errore nell'apertura del file analog_ini.txt"      ! nello script serve
          read(entrata,*, iostat=st)nstep,n_membri,iniziotrain,finetrain,iniziotest,finetest
     print*,"Sono qui dopo"
          read(entrata,*, iostat=st)numpredictor
          read(entrata,*, iostat=st)(weight(i),i=1,numpredictor)
          read(entrata,*, iostat=st)(type_predictor(i),i=1,numpredictor)
          read(entrata,*, iostat=st)threshold_bias
!          read(entrata,*, iostat=st)(namepredictor(i),i=1,numpredictor)
    close(entrata)

print*,"Start End train, Start End Test",iniziotrain,finetrain,iniziotest,finetest
print*,"Number of members",n_membri
print*,"Num. of forec. lead time",nstep
print*,"Number of predictors",numpredictor
print*,"Weight Used",weight(1:numpredictor)
print*,"type of predictors",type_predictor(1:numpredictor)
print*,"threshold_bias",threshold_bias
50 format(10f12.2)


!ste hurricane allocate (data4d((numpredictor+1),finetest,nstep),data4d_true((numpredictor+1),finetest,nstep))
allocate (errvel((finetest*nstep),3),errvel2((finetest*nstep),3),errvel1((finetest*nstep),3))
allocate (back_up(finetest*nstep),dir_grad(finetest*nstep))
allocate (eps_analog((finetest*nstep),n_membri),num_hurricane((finetest*nstep),n_membri),date_hurricane((finetest*nstep),n_membri))
allocate (data4d((numpredictor+1),finetest,nstep),data4d_true((numpredictor+1),finetest,nstep))

ndaytest=finetest-iniziotest+1
ndaytrain=finetrain-iniziotrain+1

open(entrata, file="RunAna.txt", status="old", iostat=st)
 do j= 1,finetest
    do k= 1,nstep

!       indice=(j-1)*flt+k
        read(entrata,*,iostat=st) (data4d(i,j,k),i=1,(numpredictor+1))
!        print*,"data",i,j,k,data4d(:,j,k)
      if (st > 0) stop "Errore nella lettura del file stazioni"
      if (st < 0) exit

    enddo
  enddo

i=0
   call cpu_time(start)

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
!   data4d(numpredictor+1,iday-1,8:nstep)=-9999.000  ! #future measurements are not available
!   data4d(numpredictor+1,iday-2,16:nstep)=-9999.000 ! ##
!   data4d(numpredictor+1,iday-3,24:nstep)=-9999.000 ! ##
!   data4d(numpredictor+1,iday-4,32:nstep)=-9999.000 ! ##
!   data4d(numpredictor+1,iday-5,40:nstep)=-9999.000 ! ##

      do ivar=1,numpredictor
       if(weight(ivar)>0.00001)then
        if(type_predictor(ivar)== 1)then

!sd_var(ivar)=sd(train_altpos(,namepredictor(ivar)),na.rm=TRUE)
           sd_var(ivar)=sd(data4d(ivar,1:(iday-1),ltime),(iday-1))
        else
!  dir_grad=train_altpos(,namepredictor(ivar))
           dir_grad(1:(iday-1))=data4d(ivar,1:(iday-1),ltime)
! print*,"dirgrad",dir_grad(101:189)
           sd_var(ivar)=std_dir(dir_grad(1:(iday-1)),(iday-1))
        endif
       else
        sd_var(ivar)=1. !this is the case where a variable has weight =0, just put a value of sd without computing
       endif
       if(sd_var(ivar)==0.)then
        eps_analog(i,1:n_membri)=-9999.000  !if one of the sd =0 probably night condition solar forecasting, avoid anen computation
        go to 2405
       endif
    enddo
!     print*,"sdvar",sd_var(1:4)

      ii=0
       do idaytrain=iniziotrain,finetrain
!      do idaytrain=iniziotrain,(iday-1)   !cycle on the training
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

         errvel(ii,2)=data4d(numpredictor+1,idaytrain,ltime)!associo ad ogni errore la potenza misurata alla stessa cadenza

       !  errvel(ii,5)=data4d(1,idaytrain,ltime)! questa e' la colonna delle past forecast della variabile prevista dal modello (Delta Vmax)
         errvel(ii,3)=var_prev_t(1)
3012   CONTINUE
      enddo ! end cycle on the training

     errvel1(1:(idaytrain-1),:)=errvel(1:(idaytrain-1),:)

        miss=0
        valid=0
       do iii=1,(idaytrain-1)

        if(errvel1(iii,1)<-9998.000 .or. errvel1(iii,2)<-9998.000)then
          miss=miss+1
        else
         valid=valid+1
         errvel2(valid,:)=errvel1(iii,:)
        endif

       enddo

!      call sort_matrix(errvel2,2)
        back_up=errvel2(:,1)
        call   SSORT (errvel2(:,1), errvel2(:,2), valid, 2)! order the distance matrix and consequently the analogs on the second column
        errvel2(:,1)=back_up
        call   SSORT (errvel2(:,1), errvel2(:,3), valid, 2)
        errvel2(:,1)=back_up

      if(valid <= n_membri)then ! to deal with too many missing data
        eps_analog(i,:)=-9999.000
       else
!    print*,errvel2(1:n_membri,5)
!          bias=obs_bias(var_prev(1),errvel2(1:n_membri,3),errvel2(1:n_membri,2),n_membri)!least square method to go from deltaf to deltaobs
           bias=obs_bias(var_prev(1),errvel2(1:valid,3),errvel2(1:valid,2),valid,threshold_bias,n_membri)!least square method to go from deltaf to deltaobs

!          sig_bias=sigma_bias(errvel2(1:n_membri,3),errvel2(1:n_membri,2),n_membri)!least square method to go from deltaf to deltaobs
!bias=var_prev(1)-sum(errvel2(1:n_membri,5))/n_membri  !difference between current forecast of deltaVMAX, and the mean of past analog forecasts
         do iiii=1,n_membri
  !      if(var_prev(1)>30 .or.var_prev(1)<-30)then
  !             bias=0.
               eps_analog(i,iiii)=errvel2(iiii,2)+bias!
 !            print*,"sig_bias, eps_an(i),prima",sig_bias,eps_analog(i,iiii)
!         else
   !          eps_analog(i,iiii)=errvel2(iiii,2) !standard analog
   !       endif
!          num_hurricane(i,iiii)=(errvel2(iiii,3))
!          date_hurricane(i,iiii)=(errvel2(iiii,4))
         enddo

         mean_an=sum(eps_analog(i,:))/n_membri
!         do iiii=1,n_membri
!        eps_analog(i,iiii)= mean_an+sig_bias*(eps_analog(i,iiii)-mean_an)
!    print*,"sig_bias, eps_an(i),dopo",sig_bias,eps_analog(i,iiii)
!         enddo


      endif

!    print*,"iday,itime",iday,itime
!    print*,"eps_analog",eps_analog(i,1:n_membri)
2405 CONTINUE  !!here you arrive when a missing forecast is found

    enddo !cycle itime

   enddo  !cycle iday

  call cpu_time(finish)
  print '("Time = ",f6.3," seconds.")',finish-start
uscita=54
uscita1=55
uscita3=56
open(uscita, file="eps_analog.txt",status='unknown')
do ii=1,(ndaytest*nstep)
       write(uscita,*)(eps_analog(ii,k),k=1,n_membri)

55 format('20I3')
56 format('20I8')
enddo

close(uscita)





end program analog

real  function obs_bias(cur_fc,past_fc,past_obs,n_data,threshold_bias,n_members)
implicit none
real cur_fc,m,q,ymean,xmean,deltaf,sd,threshold_bias,xmean_mem
real past_fc(n_data),past_obs(n_data),finto(n_data)
integer n_members,n_data
!call SSORT(past_fc(1:n_members) ,finto, n_members, 1)
!print*,"membri dopo",forecast(i,1:20)

 ymean=sum(past_obs(:))/n_data
 xmean=sum(past_fc(:))/n_data
 xmean_mem=sum(past_fc(1:n_members))/n_members
! xmean=sum(past_fc(10:11))/2. !!!median of the distribution with 20 sorted members 
 m=sum((past_fc(:)-xmean)*(past_obs(:)-ymean))/(sum((past_fc(:)-xmean)**2))
 q=ymean-m*xmean! intercept
 deltaf=cur_fc-xmean_mem
 if(cur_fc<threshold_bias)then
  obs_bias=0.
else
 obs_bias=m*deltaf
endif
! m=1./1.3
!  m=1.
! if(cur_fc>10.)then
! obs_bias=m*deltaf
! else
! obs_bias=0.
 !endif
!if (cur_fc>minval(past_fc) .and. cur_fc<maxval(past_fc))then
!   obs_bias=0.
! else
!   obs_bias=m*deltaf
!endif
!if (deltaf<0.5*sd(past_fc,n_members))then
!obs_bias=0.
!else
!obs_bias=m*deltaf
!endif! obs_bias=deltaf*0.
!print*,cur_fc,xmean,ymean,obs_bias
end function

real  function sigma_bias(past_fc,past_obs,n_members)
! computing the coeeficient to reduce the bias of the analog members
! An_newi=mean(Ani)+sig(Ani)/(sig(Ani)+sig(Fi))*(Ani-mean(Ani))
!Fi are the past analof forecasts, Ani correspondent past obs
implicit none
real std_forec,std_obs,sd
real past_fc(n_members),past_obs(n_members)
integer n_members

std_forec=sd(past_fc,n_members)
std_obs=sd(past_obs,n_members)
sigma_bias=std_obs/(std_obs+std_forec)
!sigma_bias=(sqrt(std_obs*std_obs-std_forec*std_forec))/std_obs
end function


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

real  function sd(var,n)
implicit none
real :: avg,variance,sum,diff
real :: var(n)
integer :: count,i,n
count=0
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

