using System;

namespace CNN.Utils
{
    /// <summary>
    /// A class for representing numbers in a format suitable automatic differentiation (AD) in reverse accumulation.
    /// The derivative calculation is input as an action.
    /// </summary>
    public class Rev
    {
        public double Magnitude;
        public double Derivative;
        public Action<double> CalculateDerivative;

        private Rev(double y, Action<double> dy)
        {
            Magnitude = y;
            Derivative = 0;
            CalculateDerivative = dy;
        }

        public Rev(double y)
        {
            Magnitude = y;
            Derivative = 0;
            CalculateDerivative = (x) => { Derivative += x; };
        }

        public static implicit operator double(Rev d) => d.Magnitude;
        public static implicit operator Rev(double d) => new Rev(d);

        public static Rev operator +(Rev lhs, Rev rhs) =>
            new Rev(lhs.Magnitude + rhs.Magnitude, dx =>
            {
                lhs.CalculateDerivative(dx);
                rhs.CalculateDerivative(dx);
            });

        public static Rev operator +(Rev lhs, double rhs) =>
            new Rev(lhs.Magnitude + rhs, dx =>
            {
                lhs.CalculateDerivative(dx);
            });

        public static Rev operator +(double lhs, Rev rhs) =>
            new Rev(lhs + rhs.Magnitude, dx =>
            {
                rhs.CalculateDerivative(dx);
            });

        public static Rev operator -(Rev lhs, Rev rhs) =>
           new Rev(lhs.Magnitude - rhs.Magnitude, dx =>
           {
               lhs.CalculateDerivative(dx);
               rhs.CalculateDerivative(-dx);
           });

        public static Rev operator -(Rev lhs, double rhs) =>
           new Rev(lhs.Magnitude - rhs, dx =>
           {
               lhs.CalculateDerivative(dx);
           });

        public static Rev operator -(double lhs, Rev rhs) =>
          new Rev(lhs - rhs.Magnitude, dx =>
          {
              rhs.CalculateDerivative(-dx);
          });

        public static Rev operator -(Rev lhs) =>
           new Rev(-lhs.Magnitude, dx =>
           {
               lhs.CalculateDerivative(-dx);
           });

        public static Rev operator *(Rev lhs, Rev rhs) =>
        new Rev(lhs.Magnitude * rhs.Magnitude,
                dx =>
                {
                    lhs.CalculateDerivative(dx * rhs.Magnitude);
                    rhs.CalculateDerivative(dx * lhs.Magnitude);
                });

        public static Rev operator *(Rev lhs, double rhs) =>
        new Rev(lhs.Magnitude * rhs,
                dx =>
                {
                    lhs.CalculateDerivative(dx * rhs);
                });

        public static Rev operator *(double lhs, Rev rhs) =>
        new Rev(lhs * rhs.Magnitude,
                dx =>
                {
                    rhs.CalculateDerivative(dx * lhs);
                });

        public static Rev operator /(Rev lhs, Rev rhs) =>
        new Rev(lhs.Magnitude / rhs.Magnitude,
                dx =>
                {
                    lhs.CalculateDerivative(dx / rhs.Magnitude);
                    rhs.CalculateDerivative(-dx * lhs.Magnitude / (rhs.Magnitude * rhs.Magnitude));
                });

        public static Rev operator /(Rev lhs, double rhs) =>
        new Rev(lhs.Magnitude / rhs,
               dx =>
               {
                   lhs.CalculateDerivative(dx / rhs);
               });

        public static Rev operator /(double lhs, Rev rhs) =>
        new Rev(lhs / rhs.Magnitude,
                dx =>
                {
                    rhs.CalculateDerivative(-dx * lhs / (rhs.Magnitude * rhs.Magnitude));
                });

        public Rev Pow(double e)
        {
            var x = Magnitude;
            var k = CalculateDerivative;
            return new Rev(Math.Pow(Magnitude, e),
                           dx => k(e * Math.Pow(x, e - 1) * dx));
        }

        public Rev Exp()
        {
            var x = Magnitude;
            var k = CalculateDerivative;
            return new Rev(Math.Exp(Magnitude),
                           dx => k(Math.Exp(x) * dx));
        }

        public Rev Log()
        {
            var x = Magnitude;
            var k = CalculateDerivative;
            return new Rev(Math.Log(Magnitude),
                           dx => k(1.0 / x * dx));
        }
       

    }

}