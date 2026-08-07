clear;
% figure('color','w')
clc;

state_video=0;%%是否生成视频
state_fluid=0;%%是否生成流场
state_p=1;%%是否生成压力场

file='D:\DNS\1110\';
% file='C:\Users\customer\Desktop\DNS_0603\';
filename=dir(fullfile(file,'*.dat'));

filename1=filename;
[~,idx]=sort([filename1.datenum],'ascend');
filename=filename1(idx);
clear idx filename1

xyz=importdata(strcat(file,'mesh.dat'));
prgrad=importdata(strcat(file,'prgrad.dat')); %%importdata('D:\DNS\prgrad_err.dat');
% wmean=importdata(strcat(file,'stat.dat'));
% xyz=importdata('C:\Users\customer\Desktop\DNS_0603\mesh.dat');

nu = 2.3310E-4;

x_length=192;
y_length=192;
z_length=128;

nyc=y_length-1;

x(:)=xyz(1:x_length);
y(:)=xyz(x_length+1:x_length+y_length);
z(:)=xyz(x_length+y_length+1:end);
[Z,X,Y]=meshgrid(z,x,y);

yc=0.5*(y(1:end-1)+y(2:end));
for k=1:nyc
  dyp(k)=y(k+1)-y(k);
end
for k=2:nyc
  dyc(k)=yc(k)-yc(k-1);
end
dyc(1)=dyp(1);
dyc(y_length)=dyp(nyc);
% urms(y_length)=(0);
%%
fmat=moviein(999);%%视频
k=1;
% ox(1:x_length,1:z_length,1:y_length)=[0];
% oy(1:x_length,1:z_length,1:y_length)=[0];
% oz(1:x_length,1:z_length,1:y_length)=[0];
% ox2(1:x_length,1:z_length,1:y_length)=[0];
% oy2(1:x_length,1:z_length,1:y_length)=[0];
% oz2(1:x_length,1:z_length,1:y_length)=[0];
ox(1,1,1:y_length)=[0];
oy(1,1,1:y_length)=[0];
oz(1,1,1:y_length)=[0];
ox2(1,1,1:y_length)=[0];
oy2(1,1,1:y_length)=[0];
oz2(1,1,1:y_length)=[0];

for i=1:1:length(filename)
tic
    if filename(i).name(1:3)=='dns'
        % flu=importdata('C:\Users\customer\Desktop\DNS_0603\dns_data5000.dat');
        flu=importdata(strcat(file,filename(i).name));
        
        % x=reshape(xyz(:,1),x_length,z_length,y_length);x1(:)=x(:,1,1);
        % y=reshape(xyz(:,2),x_length,z_length,y_length);y1(:)=y(1,1,:);
        % z=reshape(xyz(:,3),x_length,z_length,y_length);z1(:)=z(1,:,1);
        u=reshape(flu(:,1),x_length,z_length,y_length)+2/3;
        v=reshape(flu(:,2),x_length,z_length,y_length);
        w=reshape(flu(:,3),x_length,z_length,y_length);
        if state_p==1
            p=reshape(flu(:,4),x_length,z_length,y_length);
        end
        %%
        
        if state_fluid==1
            figure(1)
            set(gcf,'color','w')
            clf
            xslice = [x(160),x(90),x(2)];
            yslice = [y(4)];
            zslice = [z(100),z(2)];
            
            slice(Z,X,Y,u,zslice,xslice,yslice)
            
            
            %     [curlz,curlx,curly,~] = curl(Z,X,Y,w,u,v);
            %     slice(Z,X,Y,curly,zslice,xslice,yslice)
            
            axis equal
            %   colormap([[linspace(0,199/200,200)',linspace(0,199/200,200)',ones(200,1)];[1,1,1];[ones(200,1),linspace(199/200,0,200)',linspace(199/200,0,200)']])
            colormap jet
            caxis([0,0.95]);
            shading interp
            colorbar
            box on
            grid on
            
            
            %     hiso = patch(isosurface(z,x,y,curlx,2.5,u+2/3));
            %     isonormals(z,x,y,curlx,hiso)
            %     hiso.FaceColor = 'interp'; %等值面上色
            %     hiso.EdgeColor = 'none';
            %     set(gca,'BoxStyle','full','Box','on')
            axis tight
            
            camproj perspective
            view(-30,15)
            camlight(-45,45)
            hcap.AmbientStrength = 1;
            lighting gouraud
            daspect([1 1 1]);
            
            title(filename(i).name)
            
        end
        %%
        if state_video==1
            fmat(k)=getframe(gcf);    %截取帧
        end
        %%
        utau(k)=sqrt(nu*(mean(u(:,:,3),'all')-mean(u(:,:,2),'all'))/(y(3)-y(2)))+...
            sqrt(nu*(mean(u(:,:,y_length-2),'all')-mean(u(:,:,y_length-1),'all'))/(y(y_length-1)-y(y_length-2)));
        utau(k)=0.25*utau(k)+sqrt(abs(mean(prgrad(15000:end))))*0.5;
        Retau(k)=utau(k)*1.0/nu;
        
        umean(:,k)=mean(u,[1,2]);
        vmean(:,k)=mean(v,[1,2]);
        wmean(:,k)=mean(w,[1,2]);
        pmean(:,k)=mean(p,[1,2]);
        
        urms(:,k)=rms((u-mean(u,[1,2])),[1,2]);
        vrms(:,k)=rms((v-mean(v,[1,2])),[1,2]);
        wrms(:,k)=rms((w-mean(w,[1,2])),[1,2]);
        
        %         [curlz,curlx,curly,~] = curl(Z,X,Y,w,u,v);
        %         wxrms(:,k)=rms((curlx-mean(curlx,[1,2])),[1,2]);
        %         wyrms(:,k)=rms((curly-mean(curly,[1,2])),[1,2]);
        %         wzrms(:,k)=rms((curlz-mean(curlz,[1,2])),[1,2]);
        
        [curlz,curlx,curly,~] = curl(Z,X,Y,w,u,v);
        %         wxrms(:,k)=rms(curlx,[1,2]);
        %         wyrms(:,k)=rms(curly,[1,2]);
        %         wzrms(:,k)=rms(curlz,[1,2]);
        
%         [graduz,~,graduy]=gradient(u,1,1,y);
%         [gradvz,gradvx,~]=gradient(v,1,1,1);
%         [~,gradwx,gradwy]=gradient(w,1,1,y);
%         dudy(:,k)=mean(graduy,[1,2]);
%         dwdy(:,k)=mean(gradwy,[1,2]);
        %
        %         wxrms(:,k)=rms(gradwy-mean(gradwy,[1,2])-(gradvz-mean(gradvz,[1,2])),[1,2]);
        %         wyrms(:,k)=rms(graduz-mean(graduz,[1,2])-(gradwx-mean(gradwx,[1,2])),[1,2]);
        %         wzrms(:,k)=rms(gradvx-mean(gradvx,[1,2])-(graduy-mean(graduy,[1,2])),[1,2]);
        
%         ox=ox+mean(curlx,[1,2]);
%         oy=oy+mean(curly,[1,2]);
%         oz=oz+mean(curlz,[1,2]);
%         
%         ox2=ox2+mean(curlx.^2,[1,2]);
%         oy2=oy2+mean(curly.^2,[1,2]);
%         oz2=oz2+mean(curlz.^2,[1,2]);
        
                wxrms(:,k)=(mean(curlx.^2,[1,2])-mean(curlx,[1,2]).^2);
                wyrms(:,k)=(mean(curly.^2,[1,2])-mean(curly,[1,2]).^2);
                wzrms(:,k)=(mean(curlz.^2,[1,2])-mean(curlz,[1,2]).^2);
        
%         wxrms(:,k)=sqrt(mean(curlx.^2,[1,2])-mean(gradwy.^2,[1,2]));
%         wyrms(:,k)=mean(sqrt(curly.^2),[1,2]);
%         wzrms(:,k)=sqrt(mean(curlz.^2,[1,2])-mean(graduy.^2,[1,2]));
        
        prms(:,k)=rms((p-mean(p,[1,2])),[1,2]);
        
        for kk=2:nyc
            dudyp(kk)=(mean(u(:,:,kk),[1,2])-mean(u(:,:,kk-1),[1,2]))/dyc(kk);
        end
        kk=1;
        dudyp(kk)= 2.0*mean(u(:,:,kk),[1,2])/dyc(kk);
        kk=y_length;
        dudyp(kk)=-2.0*mean(u(:,:,nyc),[1,2])/dyc(kk);
        dudyc=0.5*(dudyp(1:end-1)+dudyp(2:end));
        
        for kk=2:nyc
            dwdyp(kk)=(mean(w(:,:,kk),[1,2])-mean(w(:,:,kk-1),[1,2]))/dyc(k);
        end
        kk=1;
        dwdyp(kk)= 2.0*mean(u(:,:,kk),[1,2])/dyc(kk);
        kk=y_length;
        dwdyp(kk)=-2.0*mean(u(:,:,nyc),[1,2])/dyc(kk);
        dwdyc=0.5*(dwdyp(1:end-1)+dwdyp(2:end));
        
        dudy(:,k)=dudyp;
        dwdy(:,k)=dwdyp;
        
        %         figure(2)
        %         hold on
        %         plot(k,Retau(k),'ro')
        %         figure(3)
        %         hold on
        %         plot(y*utau(k)/nu,urms(:,k)/utau(k))
        %         xlabel('$y^+$',"Interpreter","latex")
        %         ylabel('$u^+_{rms}$',"Interpreter","latex")
        %         figure(4)
        %         hold on
        %         plot(wmean(:,k))
        %         plot(y*utau(:,k)/nu,wxrms(:,k)/utau(:,k)/utau(:,k)*nu)
        %         figure(5)
        %         hold on
        %         plot(y*utau(:,k)/nu,prms(:,k)/utau(:,k)/utau(:,k))
        
        k=k+1;
    end
toc
disp(k)
end
%%

turb(y_length,12)=(0);   %%%% defind turbulent statistic
turbh(fix(y_length/2+1),12)=(0);
%%%%   y+ u+ du+/dy w+ p+ u+rms v+rms w+rms p+rms wx+rms wy+rms wz+rms
%%%%   1  2     3   4  5   6    7     8       9     10     11     12
%%%%
nn=11;
Utau=mean(utau(nn:end));
turb(:,1)=y*Utau/nu;
turb(:,2)=(mean(umean(:,nn:end),2))/Utau;
turb(:,3)=abs(mean(dudy(:,nn:end),2))/Utau/Utau*nu;
turb(:,4)=mean(wmean(:,nn:end),2)/Utau;
turb(:,5)=mean(pmean(:,nn:end),2)/Utau/Utau;
turb(:,6)=mean(urms(:,nn:end),2)/Utau;
turb(:,7)=mean(vrms(:,nn:end),2)/Utau;
turb(:,8)=mean(wrms(:,nn:end),2)/Utau;
turb(:,9)=mean(prms(:,nn:end),2)/Utau/Utau;
turb(:,10)=mean((wxrms(:,nn:end)),2)*nu/Utau/Utau;
turb(:,11)=mean((wyrms(:,nn:end)),2)*nu/Utau/Utau;
turb(:,12)=mean(sqrt(wzrms(:,nn:end)),2)*nu/Utau/Utau;

turbh(:,1)=turb(1:fix(y_length/2+1),1);
for ii=1:1:y_length/2+1
    turbh(ii,2:end)=(turb(ii,2:end)+turb(y_length+1-ii,2:end))*0.5;
end
% clear turb urms vrms wrms prms wxrms wyrms wzrms
figure(6)
for ii=2:1:length(turbh(1,:))
    %     figure(ii)
    subplot(3,4,ii)
    %     clf
    set(gcf,'color','w')
    plot(turbh(:,1),turbh(:,ii),'k','LineWidth',1)
    hold on
    if(ii>5 )%%&& ii<10)
        plot(ref(:,1),sqrt(ref(:,ii)),'r','LineWidth',1)
    else
        plot(ref(:,1),ref(:,ii),'r','LineWidth',1)
    end
end


% nn=2;
% plot(y*mean(utau(:,nn:end))/nu,mean(wxrms(:,nn:end)./utau(:,nn:end)./utau(:,nn:end),2)*nu,'linewidth',1)


%%
if state_video==1
    output=strcat('C:\Users\customer\Desktop\','3');
    video=VideoWriter(output);  %视频输出
    video.FrameRate=2;
    open(video)
    writeVideo(video,fmat)
    close(video)
end
%%
% clf
% utau=sqrt(nu*(mean(u(:,:,2),'all')-mean(u(:,:,1),'all'))/(y(2)-y(1)));
% Retau=utau*1.0/nu;
% umean(:)=mean(u,[1,2]);
% hold on
% plot(y*utau/nu,(umean+2/3)/utau)
% set(gca,"XScale","log")
%
% plot(ref(:,1),ref(:,2),'ro')
%
% urms(:)=rms((u-mean(u,[1,2])),[1,2]);
% hold on
% plot(y*utau/nu,urms/utau)

%% 1118
clear;clc
file='D:\DNS\Retau1000\0110\';
stat=importdata(strcat(file,'stat.dat'));
prgrad=importdata(strcat(file,'prgrad.dat'));
load(strcat('D:\DNS\Retau1000\','ref.mat'))
nu = 5.0E-5;
Utau=sqrt(abs(mean(prgrad)));
vormag=Utau*Utau/nu;
clear ox oy oz dwdyp dyc
data_num=11;
for i=1:1:length(stat)/data_num
    ox(i,:)=stat(data_num*(i-1)+1,:);
    oy(i,:)=stat(data_num*(i-1)+2,:);
    oz(i,:)=stat(data_num*(i-1)+3,:);
    um(i,:)=stat(data_num*(i-1)+4,:);
    vm(i,:)=stat(data_num*(i-1)+5,:);
    wm(i,:)=stat(data_num*(i-1)+6,:);
    pm(i,:)=stat(data_num*(i-1)+7,:);
    um2(i,:)=stat(data_num*(i-1)+8,:);
    vm2(i,:)=stat(data_num*(i-1)+9,:);
    wm2(i,:)=stat(data_num*(i-1)+10,:);
    pm2(i,:)=stat(data_num*(i-1)+11,:);
end
ox=mean(ox);
oy=mean(oy);
oz=mean(oz);

um=mean(um);
vm=mean(vm);
wm=mean(wm);
pm=mean(pm);
um2=mean(um2);
vm2=mean(vm2);
wm2=mean(wm2);
pm2=mean(pm2);

% nn=100;
% ox=mean(ox(nn:end,:));
% oy=mean(oy(nn:end,:));
% oz=mean(oz(nn:end,:));
% 
% um=mean(um(nn:end,:));
% vm=mean(vm(nn:end,:));
% wm=mean(wm(nn:end,:));
% pm=mean(pm(nn:end,:));
% um2=mean(um2(nn:end,:));
% vm2=mean(vm2(nn:end,:));
% wm2=mean(wm2(nn:end,:));
% pm2=mean(pm2(nn:end,:));

% file='D:\DNS\1110\';
xyz=importdata(strcat(file,'mesh.dat'));
x_length=576/2;
y_length=384;
z_length=576/2;
nyp=y_length-2;
nyc=nyp-1;
x(:)=xyz(1:x_length);
y(:)=xyz(x_length+1:x_length+y_length);
z(:)=xyz(x_length+y_length+1:end);

yp=y(2:end-1);
yc=0.5*(yp(1:nyc)+yp(2:nyp));

for k=1:nyc
    dyp(k)=y(k+1)-y(k);
end

yh=yc(:,1:fix(nyc/2))*Utau/nu;

dudyp=gradient(um,y(2:end-1));
dwdyp=gradient(wm,y(2:end-1));

dudyc=0.5*(dudyp(1:end-1)+dudyp(2:end));

for i=1:1:fix(nyc/2)
    turbh(i,1)=0.5*(um(:,i)+um(:,nyc+1-i))/Utau;
    turbh(i,2)=0.5*(dudyp(:,i)-dudyp(:,nyc+1-i))/vormag;
    turbh(i,3)=0.5*(wm(:,i)+wm(:,nyc+1-i))/Utau;    
    turbh(i,4)=0.5*(pm(:,i)+pm(:,nyc+1-i))/Utau/Utau;
        
    turbh(i,5)=0.5*(sqrt(um2(:,i)-um(:,i)^2)+sqrt(um2(:,nyc+1-i)-um(:,nyc+1-i)^2))/Utau;
    
    turbh(i,6)=0.25*(sqrt(vm2(:,i)-vm(:,i)^2)+sqrt(vm2(:,nyc+1-i)-vm(:,nyc+1-i)^2)+...
        sqrt(vm2(:,i+1)-vm(:,i+1)^2)+sqrt(vm2(:,nyp+1-i)-vm(:,nyp+1-i)^2))/Utau;
    turbh(i,7)=0.5*(sqrt((wm2(:,i)-wm(:,i)^2))+sqrt((wm2(:,nyc+1-i)-wm(:,nyc+1-i)^2)))/Utau;
    
    turbh(i,8)=0.5*(sqrt(pm2(:,i)-pm(:,i)^2)+sqrt(pm2(:,nyc+1-i)-pm(:,nyc+1-i)^2))/Utau/Utau;
    
    turbh(i,9)=0.25/vormag*(sqrt(ox(:,i)-dwdyp(i)^2)+sqrt(ox(:,nyc+1-i)-dwdyp(nyc+1-i)^2)+...
        sqrt(ox(:,i+1)-dwdyp(i+1)^2)+sqrt(ox(:,nyp+1-i)-dwdyp(nyp+1-i)^2));
    turbh(i,10)=0.5/vormag*(sqrt(oy(:,i))+sqrt(oy(:,nyc+1-i)));
    turbh(i,11)=0.25/vormag*(sqrt(oz(:,i)-dudyp(i)^2)+sqrt(oz(:,nyc+1-i)-dudyp(nyc+1-i)^2)+...
        sqrt(oz(:,i+1)-dudyp(i+1)^2)+sqrt(oz(:,nyp+1-i)-dudyp(nyp+1-i)^2));
end

figure(7)
% clf
for ii=1:1:length(turbh(1,:))
    subplot(3,4,ii+1)
    set(gcf,'color','w')
    plot(yh,turbh(:,ii),'r','LineWidth',1)
    hold on
    if(ii>4 )%%&& ii<10)
        plot(ref(:,1),sqrt(ref(:,ii+1)),'k.','LineWidth',1)
    else
        plot(ref(:,1),ref(:,ii+1),'k.','LineWidth',1)
    end    
    xlabel('$y^+$',"Interpreter","latex")
    set(gca,"XScale",'log')
end


ylabel('$\overline{u}$',"Interpreter","latex")
ylabel('$\mathrm{d} \overline{u}/\mathrm{d} y$',"Interpreter","latex")
ylabel('$\overline{w}$',"Interpreter","latex")
ylabel('$\overline{p}$',"Interpreter","latex")
ylabel('$u^+_{rms}$',"Interpreter","latex")
ylabel('$v^+_{rms}$',"Interpreter","latex")
ylabel('$w^+_{rms}$',"Interpreter","latex")
ylabel('$p^+_{rms}$',"Interpreter","latex")
ylabel('$\omega^+_{x,rms}$',"Interpreter","latex")
ylabel('$\omega^+_{y,rms}$',"Interpreter","latex")
ylabel('$\omega^+_{z,rms}$',"Interpreter","latex")
% figure(2);clf
% set(gcf,'color','w')
% subplot(2,2,1)
% plot(yh,oxm','k','LineWidth',1)
% subplot(2,2,2)
% plot(yh,oym','k','LineWidth',1)
% subplot(2,2,3)
% plot(yh,ozm','k','LineWidth',1)
% subplot(2,2,4)
% plot(yc,dudyc','k','LineWidth',1)