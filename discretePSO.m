clc;
clear all;
close all;

%% problem definition
data = load ('BreastCancerStroma.mat');
nVar = 22283;
SigmoidFunc = @(v) sigmoid(v,nVar);
AccuracyFunc = @(u) classificationkfold(u,data);
VarSize =[1 nVar]; 

%% Parameters

w=1;
wdamp= .79;
c1=2;
c2= 2;
MaxIt = 20;
nPop = 50;  
     
%% Initialization
empty_particle.Position=[];

empty_particle.Velocity=[];
empty_particle.Accuracy=[];
empty_particle.s=[];
empty_particle.delta=[];
empty_particle.Best.Position=[];
empty_particle.Best.Accuracy=[];

particle=repmat(empty_particle,nPop,1);

BestSol.Accuracy=0;


for i=1:nPop
      % initialize position  
      particle(i).Position = round(unifrnd(0,1,VarSize ));
      % initialize Velocity  
      particle(i).Velocity = zeros(VarSize );
      % Evaluation
      particle(i).Accuracy=AccuracyFunc(particle(i).Position);
      % Initialize Sigmoid and delta  
      particle(i).s  = zeros(VarSize); 
      particle(i).delta=zeros(VarSize);
      % Update personal best  
      particle(i).Best.Position=particle(i).Position;
      particle(i).Best.Accuracy=particle(i).Accuracy;
      % Update Global Best
      if particle(i).Best.Accuracy>BestSol.Accuracy      
        BestSol=particle(i).Best;        
      end
      
end
BestAccuracy=zeros(MaxIt,1);
%% MAIN LOOP

for it = 1:MaxIt
    for i =1:nPop
        %update velocity
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*unifrnd(0,1,VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*unifrnd(0,1,VarSize).*(BestSol.Position-particle(i).Position);
%           particle(i).Velocity = w*particle(i).Velocity ...
%             +c1*randi([0,1],VarSize).*(particle(i).Best.Position-particle(i).Position) ...
%             +c2*randi([0,1],VarSize).*(BestSol.Position-particle(i).Position);
%       
%         % Apply Velocity Limits
%         particle(i).Velocity = max(particle(i).Velocity,VelMin);
%         particle(i).Velocity = min(particle(i).Velocity,VelMax);
        % determine sigmoid and delta value
        particle(i).s = SigmoidFunc(particle(i).Velocity); 
        particle(i).delta = unifrnd(0,1,VarSize );
        % Update Position
        for k= 1:nVar
            if particle(i).delta(:,k) < particle(i).s(:,k)
                particle(i).Position(:,k) = 1;
            else particle(i).Position(:,k) =0;
            end
        end
        % Evaluation
        particle(i).Accuracy = AccuracyFunc(particle(i).Position);
        %disp([': Best Accuracy = ' num2str(particle(i).Accuracy)]);

        % Update Personal Best
        if particle(i).Accuracy>particle(i).Best.Accuracy
            
            particle(i).Best.Position=particle(i).Position;
            particle(i).Best.Accuracy=particle(i).Accuracy;
            
            
            % Update Global Best
            if particle(i).Best.Accuracy>BestSol.Accuracy
                
                BestSol=particle(i).Best;
                
            end
            
        end


    end
    BestAccuracy(it)=BestSol.Accuracy;
    
    disp(['**************Iteration ' num2str(it) ': Best Accuracy = ' num2str(BestAccuracy(it))]);
    w = w* wdamp;
end

%% Results

figure;
plot(BestAccuracy,'LineWidth',2);
%semilogy(BestAccuracy,'LineWidth',2);
xlabel('Iteration');
ylabel('Accuracy');








