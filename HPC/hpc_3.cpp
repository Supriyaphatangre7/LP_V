#include<iostream>
#include<omp.h>
#include<climits>
using namespace std;

int main()
{
    int n;
    cout<<"eneter no of ele:";
    cin>>n;

    int arr[n];

    cout<<"enter array ele:";
    for(int i=0;i<n;i++)
    {
        cin>>arr[i];
    }

    int sum=0;
    int minval=INT_MAX;
    int maxval=INT_MIN;

    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++)
    {
        sum+=arr[i];
    }

    #pragma omp parallel for reduction(min:minval)
    for(int i=0;i<n;i++)
    {
        if(arr[i]<minval)
        {
            minval=arr[i];
        }
    }

    #pragma omp parallel for reduction(max:maxval)
    for(int i=0;i<n;i++)
    {
        if(arr[i]>maxval)
        {
            maxval=arr[i];
        }
    }

    double avg=(double)sum/n;

    cout<<sum<<" "<<minval<<" "<<maxval<<" "<<avg;
    
}