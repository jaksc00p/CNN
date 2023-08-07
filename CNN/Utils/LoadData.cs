using System;

namespace CNN.Utils
{
    public static class LoadData
    {
        /// <summary>
        /// Read n images from a text file. Assume images are storead as a matrix with nx column and ny rows. 
        /// Scale all values before returning.
        /// </summary>
        public static double[,,] LoadInputData(string filename, int n, int nx, int ny, char separator, double scalefactor)
        {
            double[,,] data = new double[n, nx, ny];

            int j = 0;
            int k = 0;
            using (var reader = new StreamReader(filename))
            {
                if (reader != null)
                {
                    string line = reader.ReadLine();
                    while (line != null)
                    {
                        string[] line_split = line.Split(new char[] { separator }, StringSplitOptions.RemoveEmptyEntries);
                        if (line_split.Length != nx)
                            throw new InvalidOperationException("Error: wrong data format");
                        for (int i = 0; i < nx; i++)
                        {
                            if (double.TryParse(line_split[i], out double num))
                            {
                                data[k, j, i] = num * scalefactor;
                            }
                            else
                            {
                                throw new InvalidOperationException("Error: wrong data format");
                            }
                        }

                        if (++j >= ny)
                        {
                            j = 0;
                            k++;
                        }

                        if (k == n)
                            break;

                        line = reader.ReadLine();
                    }
                }
            }

            if (j != 0 || k != n)
                throw new InvalidOperationException("Error: wrong data format");

            return data;

        }

        /// <summary>
        /// Read n digits from a text file. Assume one digit is printed on each line.
        /// </summary>
        public static double[] LoadOutputData(string filename, int n)
        {
            double[] data = new double[n];

            int k = 0;
            using (var reader = new StreamReader(filename))
            {
                if (reader != null)
                {
                    string line = reader.ReadLine();
                    while (line != null)
                    {
                        if (double.TryParse(line, out double num))
                        {
                            data[k++] = num;
                        }
                        else
                        {
                            throw new InvalidOperationException("Error: wrong data format");
                        }

                        if (k == n)
                            break;

                        line = reader.ReadLine();
                    }
                }
            }

            if (k != n)
                throw new InvalidOperationException("Error: wrong data format");

            return data;
        }

    }
}
