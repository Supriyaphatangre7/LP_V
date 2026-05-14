#include<iostream>
#include<omp.h>
using namespace std;

void parallelbubble(int arr1[],int n)
{
    
    for(int i=0;i<n;i++)
    {
        #pragma omp parallel for

        for(int j=0;j<n-1;j+=2)
        {
            if(arr1[j]>arr1[j+1])
            {
                swap(arr1[j],arr1[j+1]);
            }
        }


        #pragma omp parallel for

        for(int j=1;j<n-1;j+=2)
        {
            if(arr1[j]>arr1[j+1])
            {
                swap(arr1[j],arr1[j+1]);
            }
        }
    }
}

void merge(int arr2[],int l,int m,int r)
{
    int n1=m-l+1;
    int n2=r-m;

    int L[n1],R[n2];

    for(int i=0;i<n1;i++)
    {
        L[i]=arr2[l+i];
    }

    for(int j=0;j<n2;j++)
    {
        R[j]=arr2[m+1+j];
    }

    int i=0,j=0,k=l;

    while(i<n1 && j<n2)
    {
        if(L[i]<=R[j])
        {
            arr2[k]=L[i];
            i++;
        }
        else
        {
            arr2[k]=R[j];
            j++;
        }
        k++;  
    }

    while(i<n1)
    {
        arr2[k]=L[i];
        i++;
        k++;
    }

    while(j<n2)
    {
        arr2[k]=R[j];
        j++;
        k++;
    }
}

void parallelmerge(int arr2[],int l,int r)
{
    if(l<r)
    {
        int m=(l+r)/2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelmerge(arr2,l,m);

            #pragma omp section
            parallelmerge(arr2,m+1,r);


        }

        merge(arr2,l,m,r);
    }
}


int main()
{
    int n;
    cout<<"enter no of ele:";
    cin>>n;

    int arr1[n],arr2[n];

    for(int i=0;i<n;i++)
    {
        cin>>arr1[i];
        arr2[i]=arr1[i];
    }

    parallelbubble(arr1,n);

    cout<<"bubble sort is:";

    for(int i=0;i<n;i++)
    {
        cout<<arr1[i];
    }

    parallelmerge(arr2,0,n-1);

    cout<<"merge sort is:";

    for(int i=0;i<n;i++)
    {
        cout<<arr2[i];
    }
}